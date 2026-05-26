/// Phase-3 OPoI: multi-model inference engine (safetensors + GGUF) via candle.
///
/// Models are loaded on demand when an AiRequest arrives and cached between
/// consecutive requests for the same model. Mining pauses during inference.
use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::gguf_file;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama::{Cache, Config, LlamaConfig, Llama};
use candle_transformers::models::quantized_llama::ModelWeights;
use candle_transformers::models::quantized_qwen2::ModelWeights as Qwen2Weights;
use std::io::{Read, Write};
use std::sync::{Mutex, OnceLock};
use tokenizers::Tokenizer;

use crate::models::{ModelFormat, ModelSpec};

const IPFS_GATEWAY: &str = "https://keryx-labs.com";
const SYSTEM_PROMPT_TINYLLAMA: &str =
    "You are a Keryx Network AI — a decentralized assistant running on GPU miners. \
     No internet access. Be concise.";

const SYSTEM_PROMPT_DEEPSEEK: &str =
    "You are an AI assistant running inside the Keryx decentralized network. \
     Keryx is a BlockDAG protocol where GPU miners execute AI inference as proof-of-work. \
     Results are published on-chain and secured via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     Always identify yourself as a Keryx Network AI. Be concise: 3-5 sentences max.";

const SYSTEM_PROMPT_LLAMA70B: &str =
    "You are a high-capability AI assistant running inside the Keryx decentralized network. \
     Keryx is a BlockDAG protocol where GPU miners execute AI inference as proof-of-work. \
     Results are published on-chain and secured via OPoI (Optimistic Proof of Inference). \
     You have no internet access — answer from training knowledge only. \
     Always identify yourself as a Keryx Network AI. Be thorough but concise.";

// ── Static engine state ──────────────────────────────────────────────────────

static SUPPORTED_SPECS: OnceLock<&'static [&'static ModelSpec]> = OnceLock::new();
static ENGINE: Mutex<Option<SlmEngine>> = Mutex::new(None);

enum ModelInner {
    Full { model: Llama, config: Config, cache_dtype: DType },
    Quantized(ModelWeights),
    QuantizedQwen2(Qwen2Weights),
}

struct SlmEngine {
    model_id: [u8; 32],
    name: &'static str,
    inner: ModelInner,
    tokenizer: Tokenizer,
    device: Device,
    eos_token_id: u32,
}

unsafe impl Send for SlmEngine {}
unsafe impl Sync for SlmEngine {}

// ── File management ──────────────────────────────────────────────────────────

fn model_dir(spec: &ModelSpec) -> std::path::PathBuf {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    exe_dir.join("models").join(spec.dir_name)
}

fn download_file(url: &str, dest: &std::path::Path) -> Result<()> {
    eprintln!("[keryx-miner] Downloading {} ...", url);
    let response = ureq::get(url)
        .call()
        .map_err(|e| anyhow!("HTTP GET {}: {}", url, e))?;
    let content_length: Option<u64> = response.header("Content-Length").and_then(|s| s.parse().ok());
    let mut reader = response.into_reader();
    let mut file = std::fs::File::create(dest)
        .with_context(|| format!("create {}", dest.display()))?;
    let mut downloaded: u64 = 0;
    let mut buf = vec![0u8; 65_536];
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 { break; }
        file.write_all(&buf[..n])?;
        downloaded += n as u64;
        if let Some(total) = content_length {
            eprint!("\r  {:.1}/{:.1} MB ({}%)   ",
                downloaded as f64 / 1_000_000.0,
                total as f64 / 1_000_000.0,
                downloaded * 100 / total);
            let _ = std::io::stderr().flush();
        }
    }
    eprintln!();
    Ok(())
}

fn ipfs_url(cid: &str) -> String {
    format!("{}/ipfs/{}", IPFS_GATEWAY, cid)
}

fn ensure_safetensors(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf, Vec<std::path::PathBuf>)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let cfg = dir.join("config.json");
    let wts: Vec<_> = spec.weight_cids.iter().enumerate().map(|(i, _)| {
        if spec.weight_cids.len() == 1 { dir.join("model.safetensors") }
        else { dir.join(format!("model-{:05}-of-{:05}.safetensors", i + 1, spec.weight_cids.len())) }
    }).collect();

    if tok.exists() && cfg.exists() && wts.iter().all(|p| p.exists()) {
        log::info!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, cfg, wts));
    }
    std::fs::create_dir_all(&dir)?;
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    if !cfg.exists() { download_file(&ipfs_url(spec.config_cid), &cfg)?; }
    for (i, (cid, path)) in spec.weight_cids.iter().zip(wts.iter()).enumerate() {
        if !path.exists() {
            if spec.weight_cids.len() > 1 { eprintln!("[keryx-miner] Shard {}/{}", i + 1, spec.weight_cids.len()); }
            download_file(&ipfs_url(cid), path)?;
        }
    }
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, cfg, wts))
}

fn ensure_gguf(spec: &ModelSpec) -> Result<(std::path::PathBuf, std::path::PathBuf)> {
    let dir = model_dir(spec);
    let tok = dir.join("tokenizer.json");
    let gguf = dir.join("model.gguf");

    if tok.exists() && gguf.exists() {
        log::info!("SlmEngine: found local model '{}' at {}", spec.name, dir.display());
        return Ok((tok, gguf));
    }
    std::fs::create_dir_all(&dir)?;
    eprintln!("\n[keryx-miner] Downloading model '{}' via IPFS. This happens once.\n", spec.name);
    if !tok.exists() { download_file(&ipfs_url(spec.tokenizer_cid), &tok)?; }
    if !gguf.exists() { download_file(&ipfs_url(spec.weight_cids[0]), &gguf)?; }
    eprintln!("[keryx-miner] Model '{}' ready.\n", spec.name);
    Ok((tok, gguf))
}

// ── Engine loading ───────────────────────────────────────────────────────────

fn load_engine(spec: &'static ModelSpec, device: Device) -> Result<SlmEngine> {
    log::info!("SlmEngine: loading '{}'…", spec.name);

    match spec.format {
        ModelFormat::Safetensors => {
            let (tok_path, cfg_path, wt_paths) = ensure_safetensors(spec)?;
            let config: LlamaConfig = serde_json::from_str(
                &std::fs::read_to_string(&cfg_path)?
            ).context("parse config.json")?;
            let config = config.into_config(false);
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let wt_refs: Vec<_> = wt_paths.iter().map(|p| p.as_path()).collect();
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&wt_refs, DType::F32, &device)
            }.map_err(|e| anyhow!("mmap weights: {}", e))?;
            let model = Llama::load(vb, &config).map_err(|e| anyhow!("build model: {}", e))?;
            let eos_token_id = tokenizer.token_to_id("</s>").unwrap_or(2);
            log::info!("SlmEngine: '{}' ready", spec.name);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::Full { model, config, cache_dtype: DType::F32 },
                tokenizer, device, eos_token_id,
            })
        }
        ModelFormat::Gguf => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            let model = ModelWeights::from_gguf(content, &mut gguf_file, &device)
                .map_err(|e| anyhow!("load gguf weights: {}", e))?;
            // LLaMA 3 / DeepSeek-R1 uses <|eot_id|> as EOS
            let eos_token_id = tokenizer.token_to_id("<|eot_id|>")
                .or_else(|| tokenizer.token_to_id("</s>"))
                .unwrap_or(128009);
            log::info!("SlmEngine: '{}' ready (eos_id={})", spec.name, eos_token_id);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::Quantized(model),
                tokenizer, device, eos_token_id,
            })
        }
        ModelFormat::GgufQwen2 => {
            let (tok_path, gguf_path) = ensure_gguf(spec)?;
            let tokenizer = Tokenizer::from_file(&tok_path)
                .map_err(|e| anyhow!("load tokenizer: {}", e))?;
            let mut gguf_file = std::fs::File::open(&gguf_path)
                .with_context(|| format!("open {}", gguf_path.display()))?;
            let content = gguf_file::Content::read(&mut gguf_file)
                .map_err(|e| anyhow!("read gguf: {}", e))?;
            let model = Qwen2Weights::from_gguf(content, &mut gguf_file, &device)
                .map_err(|e| anyhow!("load qwen2 gguf weights: {}", e))?;
            // Qwen2 / DeepSeek-R1-32B uses <|im_end|> as EOS (token 151645)
            let eos_token_id = tokenizer.token_to_id("<|im_end|>").unwrap_or(151645);
            log::info!("SlmEngine: '{}' ready (eos_id={})", spec.name, eos_token_id);
            Ok(SlmEngine {
                model_id: spec.model_id, name: spec.name,
                inner: ModelInner::QuantizedQwen2(model),
                tokenizer, device, eos_token_id,
            })
        }
    }
}

// ── Inference ────────────────────────────────────────────────────────────────

fn format_prompt(engine: &SlmEngine, prompt: &str) -> String {
    match engine.name {
        // LLaMA-3-based models (DeepSeek-R1-Distill-Llama-8B, Llama-3.3-70B)
        "deepseek-r1-8b" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        "llama-3.3-70b" => format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\
             <|start_header_id|>assistant<|end_header_id|>\n\n",
            SYSTEM_PROMPT_LLAMA70B, prompt
        ),
        // Qwen2-based models (DeepSeek-R1-Distill-Qwen-32B)
        "deepseek-r1-32b" => format!(
            "<|im_start|>system\n{}<|im_end|>\n\
             <|im_start|>user\n{}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n",
            SYSTEM_PROMPT_DEEPSEEK, prompt
        ),
        // Zephyr/TinyLlama chat template
        _ => format!(
            "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
            SYSTEM_PROMPT_TINYLLAMA, prompt
        ),
    }
}

/// Repetition penalty applied over a recent token window before sampling.
/// Breaks degenerate loops where the model repeats a phrase instead of emitting EOS
/// (common on distilled R1 models). 1.0 = disabled.
const REPEAT_PENALTY: f32 = 1.15;
const REPEAT_LAST_N: usize = 64;

fn generate(engine: &mut SlmEngine, prompt: &str, max_new_tokens: usize) -> Result<String> {
    let formatted = format_prompt(engine, prompt);
    let enc = engine.tokenizer.encode(formatted.as_str(), true)
        .map_err(|e| anyhow!("encode: {}", e))?;
    let mut all_tokens: Vec<u32> = enc.get_ids().to_vec();
    let mut generated: Vec<u32> = Vec::new();
    let mut lp = LogitsProcessor::new(42, Some(0.7), Some(0.9));
    let model_max = match engine.name {
        "deepseek-r1-8b" | "deepseek-r1-32b" => 1024,
        "llama-3.3-70b" => 512,
        _ => 2048,
    };
    let max_steps = max_new_tokens.min(model_max);

    match &mut engine.inner {
        ModelInner::Full { model, config, cache_dtype } => {
            let mut cache = Cache::new(true, *cache_dtype, config, &engine.device)
                .map_err(|e| anyhow!("create KV cache: {}", e))?;
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos, &mut cache)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if next == engine.eos_token_id { break; }
                all_tokens.push(next);
                generated.push(next);
            }
        }
        ModelInner::Quantized(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if next == engine.eos_token_id { break; }
                all_tokens.push(next);
                generated.push(next);
            }
        }
        ModelInner::QuantizedQwen2(model) => {
            for step in 0..max_steps {
                let (input_ids, pos) = if step == 0 {
                    (all_tokens.as_slice(), 0usize)
                } else {
                    let last = all_tokens.len() - 1;
                    (&all_tokens[last..], last)
                };
                let input = Tensor::new(input_ids, &engine.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| anyhow!("input tensor: {}", e))?;
                let logits = model.forward(&input, pos)
                    .map_err(|e| anyhow!("forward: {}", e))?;
                let next = sample_next(&logits, &mut lp, &all_tokens)?;
                if next == engine.eos_token_id { break; }
                all_tokens.push(next);
                generated.push(next);
            }
        }
    }

    engine.tokenizer.decode(&generated, true)
        .map(|s| s.trim().to_string())
        .map_err(|e| anyhow!("decode: {}", e))
}

fn sample_next(logits: &Tensor, lp: &mut LogitsProcessor, context: &[u32]) -> Result<u32> {
    let dims = logits.dims();
    let last = match dims.len() {
        3 => logits.narrow(1, dims[1] - 1, 1)?.squeeze(1)?.squeeze(0)?,
        2 => logits.narrow(0, dims[0] - 1, 1)?.squeeze(0)?,
        1 => logits.clone(),
        _ => return Err(anyhow!("unexpected logits shape {:?}", dims)),
    };
    // Penalize recently-generated tokens to break degenerate repetition loops.
    let last = if REPEAT_PENALTY != 1.0 && !context.is_empty() {
        let start = context.len().saturating_sub(REPEAT_LAST_N);
        let f32_logits = last.to_dtype(DType::F32).map_err(|e| anyhow!("logits dtype: {}", e))?;
        candle_transformers::utils::apply_repeat_penalty(&f32_logits, REPEAT_PENALTY, &context[start..])
            .map_err(|e| anyhow!("repeat penalty: {}", e))?
    } else {
        last
    };
    lp.sample(&last).map_err(|e| anyhow!("sample: {}", e))
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Register the set of models this miner supports. Called once at startup.
pub fn init_supported(specs: &'static [&'static ModelSpec]) {
    let _ = SUPPORTED_SPECS.set(specs);
}

/// Return the model_ids of all supported models (used for coinbase capability announcement).
pub fn loaded_model_ids() -> Vec<[u8; 32]> {
    SUPPORTED_SPECS.get()
        .map(|specs| specs.iter().map(|s| s.model_id).collect())
        .unwrap_or_default()
}

/// Load the requested model on demand (evicting a cached different model if needed),
/// then run inference. Blocking — call from `spawn_blocking`.
pub fn load_and_run_inference(model_id: &[u8; 32], prompt: &str, max_tokens: usize) -> Option<String> {
    let specs = SUPPORTED_SPECS.get()?;
    let spec = specs.iter().find(|s| &s.model_id == model_id)?;

    let mut guard = ENGINE.lock().expect("ENGINE mutex poisoned");

    let needs_load = guard.as_ref().map_or(true, |e| &e.model_id != model_id);
    if needs_load {
        if let Some(ref old) = *guard {
            log::info!("SlmEngine: evicting '{}' to load '{}'", old.name, spec.name);
        }
        *guard = None;
        let device = match Device::new_cuda(0) {
            Ok(d) => { log::info!("SlmEngine: CUDA device 0 active"); d }
            Err(e) => { log::error!("SlmEngine: CUDA unavailable ({}) — CPU fallback", e); Device::Cpu }
        };
        match load_engine(spec, device) {
            Ok(e) => { *guard = Some(e); }
            Err(e) => {
                log::error!("SlmEngine: failed to load '{}': {}", spec.name, e);
                return None;
            }
        }
    }

    let engine = guard.as_mut()?;
    match generate(engine, prompt, max_tokens) {
        Ok(text) if !text.is_empty() => Some(text),
        Ok(_) => Some("[empty response]".to_string()),
        Err(e) => {
            log::warn!("SlmEngine '{}' generate error: {}", engine.name, e);
            Some(format!("[inference error: {}]", e))
        }
    }
}
