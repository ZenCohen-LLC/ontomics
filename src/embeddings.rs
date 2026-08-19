use crate::types::Concept;
use anyhow::{anyhow, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::{Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::jina_bert::{
    BertModel as JinaBertModel, Config as JinaConfig,
};
use candle_transformers::models::modernbert::{Config as ModernConfig, ModernBert};
use candle_transformers::models::nomic_bert::{
    Config as NomicConfig, NomicBertModel,
};
use hf_hub::api::sync::ApiBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::Range;
use std::path::{Path, PathBuf};
use tokenizers::{PaddingParams, PaddingStrategy, Tokenizer, TruncationParams};

// ---------------------------------------------------------------------------
// Trait: pluggable embedding backend
// ---------------------------------------------------------------------------

pub trait EmbeddingModel: Send + Sync {
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;
    fn dim(&self) -> usize;
    fn model_id(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn detect_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        return Device::cuda_if_available(0).map_err(Into::into);
    }
    #[cfg(feature = "metal")]
    {
        return Device::new_metal(0).map_err(Into::into);
    }
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    {
        Ok(Device::Cpu)
    }
}

fn fetch_model_files(
    model_id: &str,
    cache_dir: Option<&PathBuf>,
) -> Result<(PathBuf, PathBuf, PathBuf)> {
    let mut builder = ApiBuilder::from_env();
    if let Some(dir) = cache_dir {
        builder = builder.with_cache_dir(dir.clone());
    }
    let api = builder.with_progress(true).build()?;
    let repo = api.model(model_id.to_string());
    let config_path = repo.get("config.json")?;
    let tokenizer_path = repo.get("tokenizer.json")?;
    let weights_path = repo.get("model.safetensors")?;
    Ok((config_path, tokenizer_path, weights_path))
}

fn load_tokenizer(path: &PathBuf, max_length: usize, use_padding: bool) -> Result<Tokenizer> {
    let mut tokenizer = Tokenizer::from_file(path)
        .map_err(|e| anyhow!("failed to load tokenizer: {e}"))?;
    if use_padding {
        tokenizer.with_padding(Some(PaddingParams {
            strategy: PaddingStrategy::BatchLongest,
            ..Default::default()
        }));
    }
    tokenizer
        .with_truncation(Some(TruncationParams {
            max_length,
            ..Default::default()
        }))
        .map_err(|e| anyhow!("failed to set truncation: {e}"))?;
    Ok(tokenizer)
}

/// Context length each long-context model was actually *trained* on.
///
/// `config.json` advertises an architectural maximum (`n_positions`) far above
/// the trained window (`max_trained_positions`). Feeding a transformer more
/// positions than it was trained for is both quadratically expensive and out
/// of distribution, so truncation is capped at the trained window instead.
const CODE_RANK_MAX_TOKENS: usize = 2048;
const GTE_MODERN_MAX_TOKENS: usize = 2048;
const JINA_CODE_MAX_TOKENS: usize = 2048;

/// Default ceiling on batch size per model, used when the caller supplies no
/// `resources.embedding_batch_size`. The attention budget below may shrink a
/// batch further; nothing ever raises it above this.
const BGE_SMALL_MAX_BATCH: usize = 32;
const CODE_RANK_MAX_BATCH: usize = 8;
const GTE_MODERN_MAX_BATCH: usize = 2;

/// Upper bound on `batch_size * seq_len^2` for a single forward pass.
///
/// Self-attention materialises a `[batch, heads, seq, seq]` score tensor, so
/// its cost scales with the square of the *longest* sequence in the batch —
/// and because padding uses [`PaddingStrategy::BatchLongest`], one long input
/// inflates every other row alongside it. A fixed batch of 8 whose longest
/// member reached 7,410 tokens asked for `8 * 12 * 7410^2 * 4 B` = 21 GB in a
/// single allocation and aborted the process.
///
/// Bounding the product keeps peak allocation flat regardless of how long any
/// individual input is. At 12 heads in f32 this budget is roughly 400 MB per
/// score tensor.
const MAX_BATCH_SEQ_SQ: usize = 8_500_000;

/// Group sequence lengths into contiguous batches respecting both `max_batch`
/// and [`MAX_BATCH_SEQ_SQ`].
///
/// Input order is preserved, so callers can concatenate results directly. A
/// sequence too long to share a batch is emitted alone rather than dropped —
/// truncation, not omission, is what bounds its cost.
fn plan_batches(lens: &[usize], max_batch: usize) -> Vec<Range<usize>> {
    let max_batch = max_batch.max(1);
    let mut batches = Vec::new();
    let mut start = 0usize;
    let mut longest = 0usize;
    for (i, &len) in lens.iter().enumerate() {
        let widest = longest.max(len);
        let count = i - start + 1;
        let fits = count <= max_batch
            && count.saturating_mul(widest.saturating_mul(widest))
                <= MAX_BATCH_SEQ_SQ;
        if fits {
            longest = widest;
        } else if i == start {
            // One sequence over budget by itself; nothing to split off.
            batches.push(start..i + 1);
            start = i + 1;
            longest = 0;
        } else {
            batches.push(start..i);
            start = i;
            longest = len;
        }
    }
    if start < lens.len() {
        batches.push(start..lens.len());
    }
    batches
}

/// Resolve the batch ceiling for a model.
///
/// `configured` is `resources.embedding_batch_size` when the caller supplied
/// it. It acts as a ceiling only: [`plan_batches`] may lower it for long
/// inputs, and nothing raises it. `None` falls back to the model's own
/// default.
fn batch_ceiling(configured: Option<usize>, model_default: usize) -> usize {
    configured.unwrap_or(model_default).max(1)
}

/// Token length of each text under the tokenizer's configured truncation.
///
/// Costs one extra tokenizer pass, which is negligible beside a twelve-layer
/// forward pass and is what lets batches be sized by real token counts rather
/// than a guess.
fn token_lengths(tokenizer: &Tokenizer, texts: &[String]) -> Result<Vec<usize>> {
    texts
        .iter()
        .map(|text| {
            tokenizer
                .encode(text.as_str(), true)
                .map(|encoding| encoding.get_ids().len())
                .map_err(|e| anyhow!("tokenization failed: {e}"))
        })
        .collect()
}

fn encode_batch(
    tokenizer: &Tokenizer,
    texts: &[String],
    device: &Device,
) -> Result<(Tensor, Tensor, Tensor, usize, usize)> {
    let encodings = tokenizer
        .encode_batch(texts.to_vec(), true)
        .map_err(|e| anyhow!("tokenization failed: {e}"))?;
    let batch_size = encodings.len();
    let seq_len = encodings[0].get_ids().len();
    let token_ids: Vec<u32> = encodings
        .iter()
        .flat_map(|e| e.get_ids().iter().copied())
        .collect();
    let type_ids: Vec<u32> = encodings
        .iter()
        .flat_map(|e| e.get_type_ids().iter().copied())
        .collect();
    let mask: Vec<u32> = encodings
        .iter()
        .flat_map(|e| e.get_attention_mask().iter().copied())
        .collect();
    let token_ids = Tensor::from_vec(token_ids, (batch_size, seq_len), device)?;
    let type_ids = Tensor::from_vec(type_ids, (batch_size, seq_len), device)?;
    let mask = Tensor::from_vec(mask, (batch_size, seq_len), device)?;
    Ok((token_ids, type_ids, mask, batch_size, seq_len))
}

// ---------------------------------------------------------------------------
// Weight remapping for models whose safetensors keys don't match candle's
// ---------------------------------------------------------------------------

/// Load safetensors into a HashMap and remap keys for Jina Code v2.
///
/// Candle's `jina_bert` module expects:
///   encoder.layer.N.mlp.gated_layers.weight
///   encoder.layer.N.mlp.wo.{weight,bias}
///   encoder.layer.N.mlp.layernorm.{weight,bias}
///
/// The model ships:
///   encoder.layer.N.mlp.up_gated_layer.weight
///   encoder.layer.N.mlp.down_layer.{weight,bias}
///   encoder.layer.N.layer_norm_2.{weight,bias}
///
/// Shapes are identical — pure name translation, no tensor ops.
fn load_jina_code_remapped(
    weights_path: &Path,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let tensors = candle_core::safetensors::load(weights_path, device)?;
    let mut remapped = HashMap::with_capacity(tensors.len());
    for (key, tensor) in tensors {
        let new_key = remap_jina_code_key(&key);
        remapped.insert(new_key, tensor);
    }
    Ok(remapped)
}

fn remap_jina_code_key(key: &str) -> String {
    // up_gated_layer -> gated_layers
    if key.contains(".mlp.up_gated_layer.") {
        return key.replace(".mlp.up_gated_layer.", ".mlp.gated_layers.");
    }
    // down_layer -> wo
    if key.contains(".mlp.down_layer.") {
        return key.replace(".mlp.down_layer.", ".mlp.wo.");
    }
    // encoder.layer.N.layer_norm_2.X -> encoder.layer.N.mlp.layernorm.X
    if key.starts_with("encoder.layer.") && key.contains(".layer_norm_2.") {
        return key.replace(".layer_norm_2.", ".mlp.layernorm.");
    }
    key.to_string()
}

/// Load safetensors into a HashMap and add `model.` prefix for GTE-ModernBERT.
///
/// Candle's `ModernBert::load` hardcodes `vb.pp("model.embeddings.tok_embeddings")`
/// etc., but the checkpoint ships keys without the `model.` prefix.
fn load_gte_modern_remapped(
    weights_path: &Path,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let tensors = candle_core::safetensors::load(weights_path, device)?;
    let mut remapped = HashMap::with_capacity(tensors.len());
    for (key, tensor) in tensors {
        if key.starts_with("embeddings.")
            || key.starts_with("layers.")
            || key.starts_with("final_norm.")
        {
            remapped.insert(format!("model.{key}"), tensor);
        } else {
            remapped.insert(key, tensor);
        }
    }
    Ok(remapped)
}

fn cls_pool_and_normalize(hidden: &Tensor, batch_size: usize) -> Result<Vec<Vec<f32>>> {
    let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?;
    let norms = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
    let normalized = cls.broadcast_div(&norms)?;
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        results.push(normalized.get(i)?.to_vec1::<f32>()?);
    }
    Ok(results)
}

fn mean_pool_and_normalize(
    hidden: &Tensor,
    mask: &Tensor,
    batch_size: usize,
) -> Result<Vec<Vec<f32>>> {
    let hidden_dim = hidden.dim(2)?;
    let seq_len = hidden.dim(1)?;
    let mask_f = mask.to_dtype(DType::F32)?;
    let mask_expanded = mask_f
        .unsqueeze(2)?
        .broadcast_as((batch_size, seq_len, hidden_dim))?;
    let sum_hidden = (hidden * &mask_expanded)?.sum(1)?;
    let sum_mask = mask_f
        .sum(1)?
        .unsqueeze(1)?
        .broadcast_as((batch_size, hidden_dim))?
        .clamp(1e-9, f64::MAX)?;
    let pooled = (&sum_hidden / &sum_mask)?;
    let norms = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
    let normalized = pooled.broadcast_div(&norms)?;
    let mut results = Vec::with_capacity(batch_size);
    for i in 0..batch_size {
        results.push(normalized.get(i)?.to_vec1::<f32>()?);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// 1. BGE-small-en-v1.5 (current default)
// ---------------------------------------------------------------------------

pub const BGE_SMALL_ID: &str = "BAAI/bge-small-en-v1.5";

pub(crate) struct BgeSmallModel {
    model: BertModel,
    tokenizer: Tokenizer,
    max_batch: usize,
}

impl BgeSmallModel {
    pub(crate) fn load(
        cache_dir: Option<&PathBuf>,
        max_batch: Option<usize>,
    ) -> Result<Self> {
        let device = detect_device()?;
        let (config_path, tokenizer_path, weights_path) =
            fetch_model_files(BGE_SMALL_ID, cache_dir)?;
        let config: BertConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let tokenizer = load_tokenizer(&tokenizer_path, 512, true)?;
        let weights = std::fs::read(&weights_path)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let model = BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            max_batch: batch_ceiling(max_batch, BGE_SMALL_MAX_BATCH),
        })
    }
}

impl EmbeddingModel for BgeSmallModel {
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let device = &self.model.device;
        let lens = token_lengths(&self.tokenizer, &texts)?;
        let mut all_results = Vec::with_capacity(texts.len());
        for range in plan_batches(&lens, self.max_batch) {
            let chunk = &texts[range];
            let (token_ids, type_ids, mask, batch_size, _) =
                encode_batch(&self.tokenizer, chunk, device)?;
            let output = self.model.forward(
                &token_ids, &type_ids, Some(&mask),
            )?;
            all_results.extend(cls_pool_and_normalize(&output, batch_size)?);
        }
        Ok(all_results)
    }

    fn dim(&self) -> usize {
        384
    }

    fn model_id(&self) -> &str {
        BGE_SMALL_ID
    }
}

// ---------------------------------------------------------------------------
// 2. Jina Code v2 (code-trained, 30 languages incl. Python/TS/JS/Rust)
// ---------------------------------------------------------------------------

const JINA_CODE_ID: &str = "jinaai/jina-embeddings-v2-base-code";

pub(crate) struct JinaCodeModel {
    model: JinaBertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl JinaCodeModel {
    pub(crate) fn load(cache_dir: Option<&PathBuf>) -> Result<Self> {
        let device = detect_device()?;
        let (config_path, tokenizer_path, weights_path) =
            fetch_model_files(JINA_CODE_ID, cache_dir)?;
        let config: JinaConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        // No padding — jina_bert Module::forward takes no attention mask,
        // so we embed one text at a time to avoid padding corruption.
        let tokenizer =
            load_tokenizer(&tokenizer_path, JINA_CODE_MAX_TOKENS, false)?;
        let remapped = load_jina_code_remapped(&weights_path, &device)?;
        let vb = VarBuilder::from_tensors(remapped, DType::F32, &device);
        let model = JinaBertModel::new(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
}

impl EmbeddingModel for JinaCodeModel {
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        // Embed one at a time — jina_bert Module::forward has no mask param,
        // so batched inference with padding would attend to pad tokens.
        let mut results = Vec::with_capacity(texts.len());
        for text in &texts {
            let encoding = self
                .tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow!("tokenization failed: {e}"))?;
            let ids = encoding.get_ids();
            let seq_len = ids.len();
            let token_ids =
                Tensor::from_vec(ids.to_vec(), (1, seq_len), &self.device)?;
            // Forward pass -> [1, seq_len, 768]
            let output = self.model.forward(&token_ids)?;
            // Mean pool over all tokens (no mask needed — no padding)
            let pooled = output.mean(1)?;
            // L2 normalize
            let norms = pooled.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
            let normalized = pooled.broadcast_div(&norms)?;
            results.push(normalized.squeeze(0)?.to_vec1::<f32>()?);
        }
        Ok(results)
    }

    fn dim(&self) -> usize {
        768
    }

    fn model_id(&self) -> &str {
        JINA_CODE_ID
    }
}

// ---------------------------------------------------------------------------
// 3. CodeRankEmbed (contrastive code retrieval, CoRNStack)
// ---------------------------------------------------------------------------

pub const CODE_RANK_ID: &str = "nomic-ai/CodeRankEmbed";

pub(crate) struct CodeRankModel {
    model: NomicBertModel,
    tokenizer: Tokenizer,
    device: Device,
    max_batch: usize,
}

impl CodeRankModel {
    pub(crate) fn load(
        cache_dir: Option<&PathBuf>,
        max_batch: Option<usize>,
    ) -> Result<Self> {
        let device = detect_device()?;
        let (config_path, tokenizer_path, weights_path) =
            fetch_model_files(CODE_RANK_ID, cache_dir)?;
        let config: NomicConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let tokenizer =
            load_tokenizer(&tokenizer_path, CODE_RANK_MAX_TOKENS, true)?;
        let weights = std::fs::read(&weights_path)?;
        let vb = VarBuilder::from_buffered_safetensors(weights, DType::F32, &device)?;
        let model = NomicBertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            max_batch: batch_ceiling(max_batch, CODE_RANK_MAX_BATCH),
        })
    }
}

impl EmbeddingModel for CodeRankModel {
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let lens = token_lengths(&self.tokenizer, &texts)?;
        let mut all_results = Vec::with_capacity(texts.len());
        for range in plan_batches(&lens, self.max_batch) {
            let chunk = &texts[range];
            let (token_ids, type_ids, mask, batch_size, _) =
                encode_batch(&self.tokenizer, chunk, &self.device)?;
            let output = self.model.forward(
                &token_ids, Some(&type_ids), Some(&mask),
            )?;
            all_results.extend(mean_pool_and_normalize(
                &output, &mask, batch_size,
            )?);
        }
        Ok(all_results)
    }

    fn dim(&self) -> usize {
        768
    }

    fn model_id(&self) -> &str {
        CODE_RANK_ID
    }
}

// ---------------------------------------------------------------------------
// 4. GTE-ModernBERT-Base (code-aware, CoIR 79.3)
// ---------------------------------------------------------------------------

const GTE_MODERN_ID: &str = "Alibaba-NLP/gte-modernbert-base";

pub(crate) struct GteModernBertModel {
    model: ModernBert,
    tokenizer: Tokenizer,
    device: Device,
    max_batch: usize,
}

impl GteModernBertModel {
    pub(crate) fn load(
        cache_dir: Option<&PathBuf>,
        max_batch: Option<usize>,
    ) -> Result<Self> {
        let device = detect_device()?;
        let (config_path, tokenizer_path, weights_path) =
            fetch_model_files(GTE_MODERN_ID, cache_dir)?;
        let config: ModernConfig =
            serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let tokenizer =
            load_tokenizer(&tokenizer_path, GTE_MODERN_MAX_TOKENS, true)?;
        let remapped = load_gte_modern_remapped(&weights_path, &device)?;
        let vb = VarBuilder::from_tensors(remapped, DType::F32, &device);
        let model = ModernBert::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            max_batch: batch_ceiling(max_batch, GTE_MODERN_MAX_BATCH),
        })
    }
}

impl EmbeddingModel for GteModernBertModel {
    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let lens = token_lengths(&self.tokenizer, &texts)?;
        let mut all_results = Vec::with_capacity(texts.len());
        for range in plan_batches(&lens, self.max_batch) {
            let chunk = &texts[range];
            let (token_ids, _, mask, batch_size, _) =
                encode_batch(&self.tokenizer, chunk, &self.device)?;
            let output = self.model.forward(&token_ids, &mask)?;
            all_results.extend(mean_pool_and_normalize(
                &output, &mask, batch_size,
            )?);
        }
        Ok(all_results)
    }

    fn dim(&self) -> usize {
        768
    }

    fn model_id(&self) -> &str {
        GTE_MODERN_ID
    }
}

// ---------------------------------------------------------------------------
// Factory: load model by HuggingFace ID
// ---------------------------------------------------------------------------

pub fn load_model(
    model_id: &str,
    cache_dir: Option<&PathBuf>,
) -> Result<Box<dyn EmbeddingModel>> {
    load_model_with_batch(model_id, cache_dir, None)
}

/// Load a model with an explicit batch ceiling.
///
/// `max_batch` comes from `resources.embedding_batch_size`. It bounds how many
/// texts may share one forward pass; the attention budget can lower that
/// further for long inputs but never raises it. `None` uses each model's own
/// default.
pub fn load_model_with_batch(
    model_id: &str,
    cache_dir: Option<&PathBuf>,
    max_batch: Option<usize>,
) -> Result<Box<dyn EmbeddingModel>> {
    match model_id {
        BGE_SMALL_ID => Ok(Box::new(BgeSmallModel::load(cache_dir, max_batch)?)),
        JINA_CODE_ID => Ok(Box::new(JinaCodeModel::load(cache_dir)?)),
        CODE_RANK_ID => Ok(Box::new(CodeRankModel::load(cache_dir, max_batch)?)),
        GTE_MODERN_ID => {
            Ok(Box::new(GteModernBertModel::load(cache_dir, max_batch)?))
        }
        other => Err(anyhow!(
            "unknown embedding model: {other}. Supported: {BGE_SMALL_ID}, \
             {JINA_CODE_ID}, {CODE_RANK_ID}, {GTE_MODERN_ID}"
        )),
    }
}

/// All supported embedding model IDs.
pub const SUPPORTED_MODELS: &[&str] = &[
    BGE_SMALL_ID,
    JINA_CODE_ID,
    CODE_RANK_ID,
    GTE_MODERN_ID,
];

// ---------------------------------------------------------------------------
// EmbeddingIndex (uses trait object)
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct EmbeddingIndex {
    model: Option<Box<dyn EmbeddingModel>>,
    vectors: HashMap<u64, Vec<f32>>,
    cache_dir: Option<PathBuf>,
    model_id: String,
    max_batch: Option<usize>,
}

impl EmbeddingIndex {
    /// Initialize embedding index with a specific model.
    pub fn new_with_model(model_id: &str, cache_dir: Option<PathBuf>) -> Result<Self> {
        Self::new_with_model_and_batch(model_id, cache_dir, None)
    }

    /// Initialize with a specific model and an explicit batch ceiling from
    /// `resources.embedding_batch_size`. The ceiling is remembered so a later
    /// [`EmbeddingIndex::load_model`] on a cache-deserialized index reuses it.
    pub fn new_with_model_and_batch(
        model_id: &str,
        cache_dir: Option<PathBuf>,
        max_batch: Option<usize>,
    ) -> Result<Self> {
        let model = load_model_with_batch(model_id, cache_dir.as_ref(), max_batch)?;
        Ok(Self {
            model: Some(model),
            vectors: HashMap::new(),
            cache_dir,
            model_id: model_id.to_string(),
            max_batch,
        })
    }

    /// Initialize embedding index with the default BGE-small model.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self> {
        Self::new_with_model(BGE_SMALL_ID, cache_dir)
    }

    /// Initialize with the default model and an explicit batch ceiling.
    pub fn new_with_batch(
        cache_dir: Option<PathBuf>,
        max_batch: Option<usize>,
    ) -> Result<Self> {
        Self::new_with_model_and_batch(BGE_SMALL_ID, cache_dir, max_batch)
    }

    /// Create an empty embedding index (no model, for when embeddings are disabled).
    pub fn empty() -> Self {
        Self::default()
    }
}

impl EmbeddingIndex {
    /// Load the embedding model into an index that was deserialized from
    /// cache (which has vectors but no model). Needed for runtime queries
    /// like embed_text.
    pub fn load_model(&mut self) -> Result<()> {
        if self.model.is_some() {
            return Ok(());
        }
        let id = if self.model_id.is_empty() {
            BGE_SMALL_ID
        } else {
            &self.model_id
        };
        let model =
            load_model_with_batch(id, self.cache_dir.as_ref(), self.max_batch)?;
        self.model = Some(model);
        Ok(())
    }

    /// Set the batch ceiling used by a subsequent
    /// [`EmbeddingIndex::load_model`]. Indexes restored from cache carry no
    /// model, so the ceiling has to be re-supplied before the model loads.
    pub fn set_max_batch(&mut self, max_batch: Option<usize>) {
        self.max_batch = max_batch;
    }

    /// Build the text representation for a concept embedding.
    fn concept_text(concept: &Concept) -> String {
        let mut parts = vec![concept.canonical.clone()];
        for subtoken in &concept.subtokens {
            if !parts.contains(subtoken) {
                parts.push(subtoken.clone());
            }
        }
        let example_ids: Vec<String> = concept
            .occurrences
            .iter()
            .map(|o| o.identifier.clone())
            .take(5)
            .collect();
        for id in &example_ids {
            if !parts.contains(id) {
                parts.push(id.clone());
            }
        }
        // Add individual words from docstrings (not full sentences) to
        // provide semantic context without overwhelming identifier terms.
        let mut doc_word_count = 0;
        for doc in &concept.doc_context {
            for word in doc.split_whitespace() {
                let w = word.trim_matches(|c: char| !c.is_alphanumeric())
                    .to_lowercase();
                if w.len() > 2 && !parts.contains(&w) {
                    parts.push(w);
                    doc_word_count += 1;
                    if doc_word_count >= 8 {
                        break;
                    }
                }
            }
            if doc_word_count >= 8 {
                break;
            }
        }
        parts.join(" ")
    }

    /// Embed a single concept using its canonical name + subtokens + example identifiers.
    #[allow(dead_code)]
    pub fn embed_concept(&mut self, concept: &Concept) -> Result<Vec<f32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no embedding model loaded (index is empty)"))?;

        let text = Self::concept_text(concept);
        let embeddings = model.embed(vec![text])?;
        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("embedding model returned no vectors"))?;

        self.vectors.insert(concept.id, vector.clone());
        Ok(vector)
    }

    /// Embed all concepts in a single batched model call.
    pub fn embed_concepts_batch(&mut self, concepts: &[Concept]) -> Result<()> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| anyhow!("no embedding model loaded (index is empty)"))?;

        if concepts.is_empty() {
            return Ok(());
        }

        let texts: Vec<String> = concepts
            .iter()
            .map(Self::concept_text)
            .collect();
        let all_embeddings = model.embed(texts)?;

        for (concept, vector) in concepts.iter().zip(all_embeddings) {
            self.vectors.insert(concept.id, vector);
        }
        Ok(())
    }

    /// Embed multiple texts in a single batched model call.
    /// Returns a Vec parallel to the input: one embedding per text.
    pub fn embed_texts_batch(&self, texts: &[String]) -> Option<Vec<Vec<f32>>> {
        let model = self.model.as_ref()?;
        model.embed(texts.to_vec()).ok()
    }

    /// Find top-k concepts most similar to the query by cosine similarity.
    pub fn find_similar(&self, query: &[f32], top_k: usize) -> Vec<(u64, f32)> {
        let mut scored: Vec<(u64, f32)> = self
            .vectors
            .iter()
            .map(|(&id, vec)| (id, cosine_similarity(query, vec)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    /// Embed arbitrary text. Returns None if no model is loaded.
    pub fn embed_text(&self, text: &str) -> Option<Vec<f32>> {
        let model = self.model.as_ref()?;
        model
            .embed(vec![text.to_string()])
            .ok()
            .and_then(|vecs| vecs.into_iter().next())
    }

    /// Look up a stored embedding vector by concept ID.
    pub fn get_vector(&self, concept_id: u64) -> Option<&Vec<f32>> {
        self.vectors.get(&concept_id)
    }

    /// Set the model cache directory (used after deserialization from cache).
    pub fn set_cache_dir(&mut self, dir: PathBuf) {
        self.cache_dir = Some(dir);
    }

    /// Return the set of concept IDs that already have embeddings.
    pub fn vector_ids(&self) -> std::collections::HashSet<u64> {
        self.vectors.keys().copied().collect()
    }

    /// Insert a single embedding vector.
    pub fn insert_vector(&mut self, id: u64, vector: Vec<f32>) {
        self.vectors.insert(id, vector);
    }

    /// Number of stored embedding vectors.
    pub fn nb_vectors(&self) -> usize {
        self.vectors.len()
    }
}

impl Serialize for EmbeddingIndex {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.vectors.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EmbeddingIndex {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let vectors = HashMap::deserialize(deserializer)?;
        Ok(Self {
            model: None,
            vectors,
            cache_dir: None,
            model_id: String::new(),
            // Not persisted: a restored index has no model yet. Callers pass
            // the ceiling via `set_max_batch` before `load_model`.
            max_batch: None,
        })
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_find_similar_empty() {
        let index = EmbeddingIndex::empty();
        let results = index.find_similar(&[1.0, 0.0], 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_find_similar_ordering() {
        let mut index = EmbeddingIndex::empty();
        index.vectors.insert(1, vec![1.0, 0.0, 0.0]);
        index.vectors.insert(2, vec![0.9, 0.1, 0.0]);
        index.vectors.insert(3, vec![0.0, 1.0, 0.0]);

        let results = index.find_similar(&[1.0, 0.0, 0.0], 3);
        assert_eq!(results[0].0, 1); // most similar
        assert_eq!(results[2].0, 3); // least similar
    }

    #[test]
    fn test_find_similar_top_k_truncation() {
        let mut index = EmbeddingIndex::empty();
        index.vectors.insert(1, vec![1.0, 0.0]);
        index.vectors.insert(2, vec![0.9, 0.1]);
        index.vectors.insert(3, vec![0.5, 0.5]);

        let results = index.find_similar(&[1.0, 0.0], 2);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_get_vector() {
        let mut index = EmbeddingIndex::empty();
        assert!(index.get_vector(1).is_none());

        index.vectors.insert(1, vec![1.0, 2.0, 3.0]);
        let v = index.get_vector(1).unwrap();
        assert_eq!(v, &vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_embed_concept_no_model() {
        use crate::types::{Concept, EntityType, Occurrence};
        use std::collections::HashSet;
        use std::path::PathBuf;

        let mut index = EmbeddingIndex::empty();
        let concept = Concept {
            id: 1,
            canonical: "transform".to_string(),
            subtokens: vec!["transform".to_string()],
            occurrences: vec![Occurrence {
                file: PathBuf::from("test.py"),
                line: 1,
                identifier: "spatial_transform".to_string(),
                entity_type: EntityType::Function,
            }],
            entity_types: HashSet::from([EntityType::Function]),
            embedding: None,
            cluster_id: None,
            subconcepts: Vec::new(),
            doc_context: Vec::new(),
        };

        let result = index.embed_concept(&concept);
        assert!(result.is_err());
    }

    #[test]
    fn test_serde_roundtrip() {
        let mut index = EmbeddingIndex::empty();
        index.vectors.insert(1, vec![1.0, 2.0, 3.0]);
        index.vectors.insert(2, vec![4.0, 5.0, 6.0]);

        let json = serde_json::to_string(&index).unwrap();
        let deserialized: EmbeddingIndex = serde_json::from_str(&json).unwrap();

        assert_eq!(
            deserialized.get_vector(1).unwrap(),
            &vec![1.0, 2.0, 3.0]
        );
        assert_eq!(
            deserialized.get_vector(2).unwrap(),
            &vec![4.0, 5.0, 6.0]
        );
        // Model should be None after deserialization
        assert!(deserialized.model.is_none());
    }

    #[test]
    #[ignore] // requires model download
    fn test_embed_text_produces_valid_vector() {
        let index = EmbeddingIndex::new(None).unwrap();
        let vector = index.embed_text("spatial transform").unwrap();
        assert_eq!(vector.len(), 384); // BGE-Small dimension
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01); // BGE-Small outputs are normalized
    }

    #[test]
    #[ignore] // requires model download
    fn test_similar_texts_have_higher_similarity() {
        let index = EmbeddingIndex::new(None).unwrap();
        let a = index.embed_text("spatial transform").unwrap();
        let b = index.embed_text("affine transformation").unwrap();
        let c = index.embed_text("database connection pool").unwrap();

        let sim_ab = cosine_similarity(&a, &b);
        let sim_ac = cosine_similarity(&a, &c);
        assert!(sim_ab > sim_ac); // related terms closer than unrelated
    }

    #[test]
    #[ignore] // requires model download
    fn test_embed_concepts_batch_stores_vectors() {
        use crate::types::{Concept, EntityType, Occurrence};
        use std::collections::HashSet;

        let mut index = EmbeddingIndex::new(None).unwrap();
        let concepts = vec![
            Concept {
                id: 1,
                canonical: "transform".to_string(),
                subtokens: vec!["transform".to_string()],
                occurrences: vec![Occurrence {
                    file: PathBuf::from("test.py"),
                    line: 1,
                    identifier: "spatial_transform".to_string(),
                    entity_type: EntityType::Function,
                }],
                entity_types: HashSet::from([EntityType::Function]),
                embedding: None,
                cluster_id: None,
                subconcepts: Vec::new(),
                doc_context: Vec::new(),
            },
            Concept {
                id: 2,
                canonical: "connection".to_string(),
                subtokens: vec!["connection".to_string()],
                occurrences: vec![Occurrence {
                    file: PathBuf::from("test.py"),
                    line: 10,
                    identifier: "db_connection".to_string(),
                    entity_type: EntityType::Variable,
                }],
                entity_types: HashSet::from([EntityType::Variable]),
                embedding: None,
                cluster_id: None,
                subconcepts: Vec::new(),
                doc_context: Vec::new(),
            },
        ];

        index.embed_concepts_batch(&concepts).unwrap();
        assert_eq!(index.nb_vectors(), 2);
        assert_eq!(index.get_vector(1).unwrap().len(), 384);
        assert_eq!(index.get_vector(2).unwrap().len(), 384);
    }

    #[test]
    fn test_supported_models_list() {
        assert_eq!(SUPPORTED_MODELS.len(), 4);
        assert!(SUPPORTED_MODELS.contains(&BGE_SMALL_ID));
        assert!(SUPPORTED_MODELS.contains(&JINA_CODE_ID));
        assert!(SUPPORTED_MODELS.contains(&CODE_RANK_ID));
        assert!(SUPPORTED_MODELS.contains(&GTE_MODERN_ID));
    }

    /// Peak attention allocation for a batch, in bytes, at 12 heads in f32.
    fn attention_bytes(count: usize, widest: usize) -> usize {
        count * 12 * widest * widest * 4
    }

    #[test]
    fn test_plan_batches_fills_to_max_batch_when_short() {
        let lens = vec![64; 20];
        let batches = plan_batches(&lens, CODE_RANK_MAX_BATCH);
        assert_eq!(batches, vec![0..8, 8..16, 16..20]);
    }

    #[test]
    fn test_plan_batches_preserves_order_and_covers_every_input() {
        let lens = vec![1500, 40, 40, 2048, 40, 900, 900, 40];
        let batches = plan_batches(&lens, CODE_RANK_MAX_BATCH);
        let mut covered = Vec::new();
        for range in &batches {
            assert!(!range.is_empty(), "empty batch: {range:?}");
            covered.extend(range.clone());
        }
        assert_eq!(covered, (0..lens.len()).collect::<Vec<_>>());
    }

    #[test]
    fn test_plan_batches_isolates_long_sequences() {
        // A long body must not drag seven short ones up to its padded width.
        let mut lens = vec![32; 7];
        lens.push(2048);
        let batches = plan_batches(&lens, CODE_RANK_MAX_BATCH);
        assert_eq!(batches, vec![0..7, 7..8]);
    }

    #[test]
    fn test_plan_batches_respects_attention_budget() {
        // Lengths spanning the full truncation range, worst-case ordering.
        let lens = vec![2048, 2048, 1024, 512, 2048, 64, 64, 1024, 2048];
        for range in plan_batches(&lens, CODE_RANK_MAX_BATCH) {
            let count = range.len();
            let widest = lens[range.clone()].iter().copied().max().unwrap_or(0);
            assert!(
                count * widest * widest <= MAX_BATCH_SEQ_SQ,
                "batch {range:?} exceeds budget: {count} x {widest}^2",
            );
            assert!(
                attention_bytes(count, widest) < 512 * 1024 * 1024,
                "batch {range:?} would allocate {} MB",
                attention_bytes(count, widest) / (1024 * 1024),
            );
        }
    }

    #[test]
    fn test_plan_batches_bounds_the_reported_regression() {
        // The failure was a batch of 8 padded to 7,410 tokens asking for 21 GB.
        // Truncation caps width; batching caps how many share that width.
        let lens = vec![CODE_RANK_MAX_TOKENS; 8];
        let batches = plan_batches(&lens, CODE_RANK_MAX_BATCH);
        assert!(batches.len() > 1, "long sequences must not share one batch");
        let worst = batches
            .iter()
            .map(|r| attention_bytes(r.len(), CODE_RANK_MAX_TOKENS))
            .max()
            .unwrap_or(0);
        assert!(
            worst < 1024 * 1024 * 1024,
            "worst batch still allocates {} MB",
            worst / (1024 * 1024),
        );
    }

    #[test]
    fn test_plan_batches_emits_oversized_sequence_alone() {
        // Defence in depth: even above the budget nothing is dropped.
        let lens = vec![64, 100_000, 64];
        let batches = plan_batches(&lens, CODE_RANK_MAX_BATCH);
        assert_eq!(batches, vec![0..1, 1..2, 2..3]);
    }

    #[test]
    fn test_plan_batches_handles_empty_and_degenerate_input() {
        assert!(plan_batches(&[], CODE_RANK_MAX_BATCH).is_empty());
        assert_eq!(plan_batches(&[10], 0), vec![0..1]);
    }

    #[test]
    fn test_batch_ceiling_prefers_configured_value() {
        assert_eq!(batch_ceiling(Some(64), CODE_RANK_MAX_BATCH), 64);
        assert_eq!(batch_ceiling(Some(2), CODE_RANK_MAX_BATCH), 2);
    }

    #[test]
    fn test_batch_ceiling_falls_back_to_model_default() {
        assert_eq!(batch_ceiling(None, CODE_RANK_MAX_BATCH), CODE_RANK_MAX_BATCH);
        assert_eq!(batch_ceiling(None, BGE_SMALL_MAX_BATCH), BGE_SMALL_MAX_BATCH);
    }

    #[test]
    fn test_batch_ceiling_never_returns_zero() {
        // config.rs clamps 0 to 1, but the model must not divide by zero even
        // if it is constructed directly.
        assert_eq!(batch_ceiling(Some(0), CODE_RANK_MAX_BATCH), 1);
        assert_eq!(batch_ceiling(None, 0), 1);
    }

    #[test]
    fn test_budget_lowers_a_generous_configured_ceiling() {
        // A caller asking for 64 still gets small batches when inputs are long:
        // the ceiling may only lower the batch, never raise memory use.
        let lens = vec![2048; 16];
        for range in plan_batches(&lens, 64) {
            assert!(
                range.len() * 2048 * 2048 <= MAX_BATCH_SEQ_SQ,
                "budget ignored for batch {range:?}",
            );
            assert!(
                attention_bytes(range.len(), 2048) < 512 * 1024 * 1024,
                "batch {range:?} allocates {} MB",
                attention_bytes(range.len(), 2048) / (1024 * 1024),
            );
        }
    }

    #[test]
    fn test_configured_ceiling_binds_when_below_budget() {
        // Short inputs would fit far more per batch, but the ceiling wins.
        let lens = vec![16; 40];
        for range in plan_batches(&lens, 4) {
            assert!(range.len() <= 4, "ceiling exceeded by {range:?}");
        }
    }

    #[test]
    fn test_trained_context_caps_stay_within_budget_alone() {
        for max_tokens in
            [CODE_RANK_MAX_TOKENS, GTE_MODERN_MAX_TOKENS, JINA_CODE_MAX_TOKENS]
        {
            assert!(
                max_tokens * max_tokens <= MAX_BATCH_SEQ_SQ,
                "a single {max_tokens}-token sequence exceeds the budget",
            );
        }
    }
}
