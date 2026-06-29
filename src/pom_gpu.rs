//! Proof-of-Model GPU mining — runs the `pom_mine` kernel in candle's CUDA context over the
//! resident weight blob to find a winning nonce. Foundation for the live mining loop (§6/3b).
//!
//! Loads the mining tier's GGUF raw (so we get per-tensor device pointers for the gather, like
//! `pom-q4-probe`) and builds the chunk-prefix gather index on the GPU. NOTE: this is a second
//! VRAM copy of the model (the inference engine holds its own). Fine for small tiers on the
//! testnet; the big tiers will share buffers later.
//!
//! The kernel's seed/pow folds are byte-identical to `pom::pom_block_seed`/`pom::pom_pow_value`,
//! so a nonce found here builds a `PomProof` (host) the node accepts.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use log::info;

use candle_core::cuda_backend::cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{CudaDevice, Device};

const PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/pom_mine.ptx"));
const CHUNK_BYTES: usize = 32;

fn words4(b: &[u8; 32]) -> [u64; 4] {
    let mut w = [0u64; 4];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

/// Magic numbers for bit-exact unsigned modulo by a runtime-constant divisor. The kernel
/// computes `q = mulhi64(a, magic) >> shift` (= `(a*magic) >> 64 >> shift`, the high half of the
/// 128-bit product, then a shift), and `rem = a - q*d == a % d`. This replaces a ~30-instruction
/// microcoded `rem.u64` on the walk's critical path with one `mul.hi` + shifts. `magic == 0` is a
/// sentinel telling the kernel to fall back to plain `%` (used when d<=1 or self-test fails).
struct ModMagic { magic: u64, shift: u32 }

/// High half of a 64x64->128 multiply — mirrors the kernel's `mulhi64` (inline PTX mul.hi.u64).
/// Kept in lockstep with cuda/pom_mine.cu so the host self-test exercises the exact same formula.
#[inline]
fn mulhi64(a: u64, b: u64) -> u64 {
    ((a as u128) * (b as u128) >> 64) as u64
}

/// Find the smallest `(magic, shift)` with `q = mulhi64(a, magic) >> shift == a / d` for all a,
/// so `rem = a - q*d == a % d` (Hacker's Delight 10-9 / libdivide branchfree, unsigned). Scans
/// `extra` from 0 so the shift is minimal; the candidate is then re-checked authoritatively by
/// `verify_mod_magic` before use. Returns magic=0 (→ kernel uses plain `%`) if no 64-bit-fitting
/// candidate is found in the scan — correctness always preserved, just no speedup in that case.
fn mod_magic(d: u64) -> ModMagic {
    if d <= 1 {
        return ModMagic { magic: 0, shift: 0 };
    }
    let p = 64 - d.leading_zeros(); // floor(log2(d))
    // Try increasing shift = (p-1) + extra; magic = ceil(2^(64+shift) / d). The smallest shift
    // whose magic fits in u64 AND verifies is the answer.
    for extra in 0..64u32 {
        let shift = (p - 1) + extra;
        // ceil(2^(64+shift) / d): 64+shift <= 127 here (shift <= 63), so the u128 numerator fits.
        let num = ((1u128 << (64 + shift)) + d as u128 - 1) / d as u128;
        if num > u64::MAX as u128 {
            continue; // magic overflows u64 at this shift; try a larger shift
        }
        let candidate = ModMagic { magic: num as u64, shift };
        if verify_mod_magic(&candidate, d) {
            return candidate; // authoritative — re-derive below before trusting
        }
    }
    ModMagic { magic: 0, shift: 0 } // no fit → fall back to plain %
}

/// Authoritative self-test: checks `fast_mod` (= the kernel's exact formula) against native `%`
/// over random + structured inputs. The node re-walks the winning nonce with native `%`, so the
/// GPU result MUST be bit-identical; this guard refuses the fast path unless it matches every
/// sampled value. Returns false → caller falls back to `%`.
fn verify_mod_magic(m: &ModMagic, d: u64) -> bool {
    if m.magic == 0 || m.shift > 63 { return false; }
    // The kernel's exact reduction: q = mulhi64(a, magic) >> shift; rem = a - q*d.
    let check = |a: u64| -> bool {
        let q = mulhi64(a, m.magic) >> m.shift;
        a.wrapping_sub(q.wrapping_mul(d)) == a % d
    };
    // Deterministic splitmix64 PRNG — reproducible on any failure.
    let mut rng = 0u64;
    for _ in 0..1_000_000 {
        rng = rng.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = rng;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        if !check(z) { return false; }
    }
    // Structured edge cases that expose off-by-one in magic constants.
    for &z in &[0u64, 1, d - 1, d, d + 1, u64::MAX,
                d.wrapping_mul(2).wrapping_sub(1), d.wrapping_mul(2),
                d.wrapping_mul(2).wrapping_add(1), d.wrapping_mul(3)] {
        if !check(z) { return false; }
    }
    true
}


pub struct PomGpuMiner {
    cuda: CudaDevice,
    stream: Arc<CudaStream>,
    bases_dev: CudaSlice<u64>,
    prefix_dev: CudaSlice<u32>,
    t_count: u32,
    n_total_chunks: u64,
    _tensors: Vec<QTensor>, // raw-loaded tensors kept alive so the gather pointers stay valid
    _shared: Vec<Arc<QTensor>>, // shared-with-inference tensors kept alive (zero-dup, Option C)
}

impl PomGpuMiner {
    /// Load the mining model's GGUF into candle (the given CUDA device ordinal), build the gather
    /// index, load the kernel. `device_id` is the CUDA ordinal the worker is bound to.
    pub fn load(gguf_path: &str, device_id: usize) -> candle_core::Result<Self> {
        let device = Device::new_cuda(device_id)?;
        let cuda = match &device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: not a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — matches pom-rt-builder / the node R_T

        let mut tensors: Vec<QTensor> = Vec::with_capacity(names.len());
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u32> = vec![0];
        for name in &names {
            let qt = content.tensor(&mut file, name, &device)?;
            let chunks = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u32;
            if chunks == 0 {
                tensors.push(qt);
                continue;
            }
            bases.push(qt.device_ptr()? as usize as u64);
            prefix.push(prefix.last().unwrap() + chunks);
            tensors.push(qt);
        }
        let n_total_chunks = *prefix.last().unwrap() as u64;
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: model produced 0 chunks".into()));
        }

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        // Warm the module cache so mine() never compiles on the hot path.
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: tensors, _shared: Vec::new() })
    }

    /// Zero-dup load (Option C): build the gather over the SAME canonical name-sorted layout as
    /// `R_T`, but for each tensor reuse the inference engine's resident VRAM buffer when it holds
    /// it quantized (`shared`, the big matrices) instead of loading a second copy. Only the
    /// dequantized-in-inference tensors (token_embd, norms) are read raw here — small. `device`
    /// MUST be the same candle device the `shared` tensors live on (pointers are context-bound).
    pub fn load_shared(
        gguf_path: &str,
        device: &Device,
        shared: &std::collections::HashMap<String, Arc<QTensor>>,
    ) -> candle_core::Result<Self> {
        let cuda = match device {
            Device::Cuda(c) => c.clone(),
            _ => return Err(candle_core::Error::Msg("PoM GPU: shared load requires a CUDA device".into())),
        };
        let stream = cuda.cuda_stream();

        let mut file = std::fs::File::open(gguf_path).map_err(candle_core::Error::wrap)?;
        let content = gguf_file::Content::read(&mut file)?;
        let mut names: Vec<String> = content.tensor_infos.keys().cloned().collect();
        names.sort(); // canonical order — must match pom-rt-builder / the node R_T

        let mut raw: Vec<QTensor> = Vec::new();
        let mut kept_shared: Vec<Arc<QTensor>> = Vec::new();
        let mut bases: Vec<u64> = Vec::new();
        let mut prefix: Vec<u32> = vec![0];
        let mut shared_hits = 0usize;
        for name in &names {
            let (ptr, chunks) = if let Some(qt) = shared.get(name) {
                // Matrix already resident for inference → reuse its buffer (zero dup).
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u32;
                let p = qt.device_ptr()? as usize as u64;
                kept_shared.push(qt.clone());
                shared_hits += 1;
                (p, c)
            } else {
                // Dequantized-in-inference (token_embd, norms): read the raw quantized bytes.
                let qt = content.tensor(&mut file, name, device)?;
                let c = (qt.storage_size_in_bytes() / CHUNK_BYTES) as u32;
                if c == 0 {
                    raw.push(qt);
                    continue;
                }
                let p = qt.device_ptr()? as usize as u64;
                raw.push(qt);
                (p, c)
            };
            if chunks == 0 {
                continue;
            }
            bases.push(ptr);
            prefix.push(prefix.last().unwrap() + chunks);
        }
        let n_total_chunks = *prefix.last().unwrap() as u64;
        if n_total_chunks == 0 {
            return Err(candle_core::Error::Msg("PoM GPU: shared load produced 0 chunks".into()));
        }
        info!("PoM zero-dup gather: {} shared tensors, {} raw-loaded, N={} chunks", shared_hits, raw.len(), n_total_chunks);

        let bases_dev = stream.clone_htod(&bases).map_err(candle_core::Error::wrap)?;
        let prefix_dev = stream.clone_htod(&prefix).map_err(candle_core::Error::wrap)?;
        let _ = cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?;

        Ok(Self { cuda, stream, bases_dev, prefix_dev, t_count: bases.len() as u32, n_total_chunks, _tensors: raw, _shared: kept_shared })
    }

    pub fn n_chunks(&self) -> u64 {
        self.n_total_chunks
    }

    /// Search nonces in `[start, start + batch)`. Returns the lowest nonce whose `pom_pow_value`
    /// is `<= target_le`, or None. `target_le` is the header's compact target as 32 LE bytes.
    pub fn mine(&self, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> candle_core::Result<Option<u64>> {
        let p = words4(pre_pow_hash);
        let t = words4(target_le);
        let k = crate::pom::POM_WALK_STEPS;
        // Precompute the magic-number modulo for n_total_chunks once per launch. The kernel uses
        // it to replace the per-step `rem.u64` (a ~30-instr microcoded divide) with a mulhi+shift.
        // mod_magic() authoritatively self-tests against native `%` for bit-exactness and returns
        // magic=0 if no valid constant exists (→ kernel falls back to plain %, correctness kept).
        let m = mod_magic(self.n_total_chunks);
        if m.magic == 0 {
            log::warn!("PoM: no magic-number modulo for n_chunks={}, using plain %", self.n_total_chunks);
        }
        let (magic, shift) = (m.magic, m.shift);
        let winner = self.stream.clone_htod(&[u64::MAX]).map_err(candle_core::Error::wrap)?;
        let grid = ((batch + 127) / 128) as u32;
        // The P40 kernel cooperatively loads prefix[0..T] into dynamic shared memory (smem
        // latency ~20 cyc beats ~400-cyc global for the per-step binary search). u32 prefix means
        // half the smem of u64 — 2 KB vs 4 KB for 501 tensors — raising P40 occupancy from 75 %
        // (48 warps/SM) to 100 % (64 warps/SM), giving the scheduler more warps to hide latency.
        let smem_bytes = ((self.t_count as usize + 1) * std::mem::size_of::<u32>()) as u32;
        let cfg = LaunchConfig { grid_dim: (grid, 1, 1), block_dim: (128, 1, 1), shared_mem_bytes: smem_bytes };

        let func = self.cuda.get_or_load_custom_func("pom_mine", "pom_mine_mod", PTX)?; // cached
        let mut b = func.builder();
        b.arg(&self.bases_dev).arg(&self.prefix_dev).arg(&self.t_count).arg(&self.n_total_chunks).arg(&k)
            .arg(&magic).arg(&shift)
            .arg(&p[0]).arg(&p[1]).arg(&p[2]).arg(&p[3]).arg(&timestamp)
            .arg(&t[0]).arg(&t[1]).arg(&t[2]).arg(&t[3])
            .arg(&start).arg(&batch).arg(&winner);
        unsafe { b.launch(cfg).map_err(candle_core::Error::wrap)?; }
        self.stream.synchronize().map_err(candle_core::Error::wrap)?;

        let w = self.stream.clone_dtoh(&winner).map_err(candle_core::Error::wrap)?[0];
        Ok(if w == u64::MAX { None } else { Some(w) })
    }
}

// The GPU miner instances, one per CUDA device ordinal. A single physical GPU may run multiple
// mining workers (PR #9: split the card's VRAM into multiple `--light` pools), each bound to the
// same device_id but driving its own PoM kernel over its own resident weight blob. Each entry can
// be individually uninstalled to free its VRAM when an inference for another model needs the GPU
// (inference has priority over PoW), then reinstalled when that worker's mining resumes.
//
// `MINERS` holds one entry per device ordinal; `INDEX_BUILD_LOCK` serializes the heavy first-time
// possession-index build so two workers on the same device don't duplicate the GGUF walk.
fn miners() -> &'static Mutex<HashMap<u32, Arc<PomGpuMiner>>> {
    static MINERS: OnceLock<Mutex<HashMap<u32, Arc<PomGpuMiner>>>> = OnceLock::new();
    MINERS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Serialize the heavy one-time possession-index build across workers (esp. on the same GPU).
fn index_build_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Install the GPU miner for `device_id` (after loading/sharing the mining model's resident
/// weights). Replaces any prior miner on that device.
pub fn install(device_id: u32, m: PomGpuMiner) {
    if let Ok(mut g) = miners().lock() {
        g.insert(device_id, Arc::new(m));
    }
}

/// Drop the GPU miners on ALL devices, releasing their hold on the mining model's VRAM (shared
/// Arcs + gather) so the inference engine can load another model. Mining is paused during
/// inference anyway. Device-agnostic by design: inference has priority over every worker.
pub fn uninstall() {
    if let Ok(mut g) = miners().lock() {
        g.clear();
    }
}

/// Whether the GPU miner for `device_id` is currently installed.
pub fn is_installed(device_id: u32) -> bool {
    miners().lock().map(|g| g.contains_key(&device_id)).unwrap_or(false)
}

/// Counts GPU miners being (re)built — each a heavy one-time model load that blocks its worker.
/// The PoW stall watchdog treats a nonzero count like an inference pause, not a crash. Counter
/// (not bool) so concurrent loads from multiple workers/devices don't clobber each other.
static LOADING: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Whether any PoM model load/rebuild is in progress (a worker is intentionally paused, not stalled).
pub fn is_loading() -> bool {
    LOADING.load(std::sync::atomic::Ordering::Relaxed) > 0
}

/// Convenience: search a nonce batch via the installed miner on `device_id`. None if that device
/// has no installed miner or no winner was found.
pub fn mine(device_id: u32, pre_pow_hash: &[u8; 32], timestamp: u64, target_le: &[u8; 32], start: u64, batch: u64) -> Option<u64> {
    let g = miners().lock().ok()?;
    let m = g.get(&device_id)?;
    m.mine(pre_pow_hash, timestamp, target_le, start, batch).ok().flatten()
}

/// Mining-tier identity for rebuilds: (model_id, gguf_path). Set once at startup.
static MINING_TIER: OnceLock<([u8; 32], String)> = OnceLock::new();

/// Record the mining tier so the miner can be rebuilt after an inference swapped the model away.
pub fn set_mining_tier(model_id: [u8; 32], gguf_path: String) {
    let _ = MINING_TIER.set((model_id, gguf_path));
}

/// Ensure the GPU miner for `device_id` is installed; if an inference evicted the mining model,
/// reload it (resident again) and rebuild the zero-dup gather. Heavy (model reload) but only when
/// needed — inference has priority, so mining reloads its model when it next gets the GPU. Returns
/// true if the miner is ready to mine. `daa` is the current block's score — used to compute the
/// PoM tier index (DAA-gated at the very-light H2 hardfork) when building the possession index.
pub fn ensure_installed(device_id: u32, daa: u64) -> bool {
    if is_installed(device_id) {
        return true;
    }
    // Flag the heavy load so the stall watchdog stays benign while the worker is blocked here.
    // Counter (not bool) so concurrent loads from multiple workers don't clobber each other.
    LOADING.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let ok = ensure_installed_inner(device_id, daa);
    LOADING.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    ok
}

/// PoM tier index of the mining model at a given block DAA. Recomputed per block (not frozen at
/// index-build time) so the tier reindexing at the very-light hardfork (H2) is applied at the
/// exact boundary — e.g. Gemma 0→1 — rather than from a stale build-time value.
pub fn current_tier(daa: u64) -> Option<u8> {
    let (model_id, _) = MINING_TIER.get()?;
    crate::models::pom_tier_index(model_id, daa)
}

/// CUDA ordinal of a candle device (None if not CUDA) — used to check whether the inference
/// engine's resident model lives on the same GPU as the PoM miner we're about to install, before
/// sharing its tensors in place.
fn cuda_gpu_id(d: &Device) -> Option<usize> {
    match d.location() {
        candle_core::DeviceLocation::Cuda { gpu_id } => Some(gpu_id),
        _ => None,
    }
}

fn ensure_installed_inner(device_id: u32, daa: u64) -> bool {
    let (model_id, gguf) = match MINING_TIER.get() {
        Some(x) => x,
        None => return false,
    };
    // Build the possession index once (host, heavy) the first time PoM activates — deferred from
    // boot so the pre-PoM legacy phase starts immediately and keeps host/GPU free. Serialized so
    // multiple workers (esp. on the same GPU) don't duplicate the GGUF walk.
    if crate::pom::active_index().is_none() {
        let _lk = index_build_lock().lock().ok();
        if crate::pom::active_index().is_none() {
            let tier = match crate::models::pom_tier_index(model_id, daa) {
                Some(t) => t,
                None => return false,
            };
            info!("PoM: building possession index (first PoM activation) — this can take a while…");
            match crate::pom::WeightIndex::build_from_gguf(gguf) {
                Ok(idx) => {
                    info!("PoM: weight index ready — N={} chunks", idx.n_chunks);
                    crate::pom::set_index(idx, tier);
                }
                Err(e) => {
                    log::error!("PoM: index build failed: {}", e);
                    return false;
                }
            }
        }
    }
    // One CUDA-resident PoM worker per GPU. This avoids all workers contending for a single
    // GPU0-bound miner object while still sharing the host-side index across the process.
    //
    // Zero-dup on the inference GPU: if the inference engine holds THIS exact model resident on
    // THIS device (split loader + `pom_force_split`), the walk shares its quantized tensors in
    // place (`load_shared`) rather than loading a second full VRAM copy — saving ~one model's
    // worth of VRAM on the serving GPU. Mining-only GPUs (no resident inference model to share)
    // fall back to a standalone copy. The N-guard below validates the gather against the host
    // index on every path, so a mismatch refuses to mine rather than producing bad proofs.
    let m = match crate::slm::pom_shared(model_id) {
        Some((inf_dev, shared)) if cuda_gpu_id(&inf_dev) == Some(device_id as usize) => {
            info!("PoM[gpu{}]: zero-dup — sharing the inference engine's resident weights (no 2nd VRAM copy)", device_id);
            PomGpuMiner::load_shared(gguf, &inf_dev, &shared)
        }
        _ => PomGpuMiner::load(gguf, device_id as usize),
    };
    match m {
        Ok(gm) => {
            let n = gm.n_chunks();
            // N-guard: the gather must match the host index, else blocks would be rejected.
            if let Some((idx, _)) = crate::pom::active_index() {
                if n != idx.n_chunks {
                    log::error!("PoM: gather N={} != index N={} — refusing to mine (rejected blocks)", n, idx.n_chunks);
                    return false;
                }
            }
            install(device_id, gm);
            info!("PoM: GPU miner ready on device {} — N={} chunks resident (matches index)", device_id, n);
            true
        }
        Err(e) => {
            log::error!("PoM: rebuild failed: {}", e);
            false
        }
    }
}
