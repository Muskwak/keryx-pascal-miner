//! Proof-of-Model — miner-side possession proof builder (build order §6).
//!
//! Byte-exact mirror of the node's verifier (`keryx-node-hardfork consensus/core/src/pom.rs`)
//! and the canonical reference (`pom-core`). The miner runs the memory-hard walk over its
//! resident weight blob; once a winning nonce is found, `build_proof` re-walks (recording the
//! trace), commits it, and opens the `t` Fiat-Shamir-selected steps with Merkle paths to the
//! tier root `R_T` and the trace root.
//!
//! The `PomProof`/`PomOpening` structs MUST keep the exact field order/types of the node's
//! (borsh wire format), and the primitives MUST stay bit-identical (the node re-derives the
//! same challenges and recomputes the same transitions). See POM_CONSENSUS_SPEC.md.

use borsh::{BorshDeserialize, BorshSerialize};

pub const CHUNK_WORDS: usize = 4; // 32 B chunk
const SEED_SALT: u64 = 0x4B65727978500; // "KeryxP"

// --- wire structs (field order == node's PomOpening/PomProof) ---

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomOpening {
    pub state_before: u64,
    pub chunk: [u8; 32],
    pub weight_path: Vec<[u8; 32]>,
    pub trace_path_before: Vec<[u8; 32]>,
    pub trace_path_after: Vec<[u8; 32]>,
}

#[derive(Clone, Debug, BorshSerialize, BorshDeserialize)]
pub struct PomProof {
    pub tier: u8,
    pub trace_root: [u8; 32],
    pub pow_value: [u8; 32],
    pub final_state: u64,
    pub initial_trace_path: Vec<[u8; 32]>,
    pub final_trace_path: Vec<[u8; 32]>,
    pub openings: Vec<PomOpening>,
}

// --- byte-exact primitives (mirror node) ---

#[inline]
pub fn blake(bytes: &[u8]) -> [u8; 32] {
    *blake3::hash(bytes).as_bytes()
}

#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d049bb133111eb);
    x ^= x >> 31;
    x
}

#[inline]
pub fn seed_state(pow_seed: u64) -> u64 {
    mix64(pow_seed ^ SEED_SALT)
}

#[inline]
pub fn transition(state: u64, chunk: &[u64; CHUNK_WORDS]) -> u64 {
    let mut h = state;
    for &w in chunk.iter() {
        h ^= w;
    }
    mix64(h)
}

#[inline]
pub fn chunk_to_words(c: &[u8; 32]) -> [u64; CHUNK_WORDS] {
    let mut w = [0u64; CHUNK_WORDS];
    for (i, wi) in w.iter_mut().enumerate() {
        *wi = u64::from_le_bytes(c[i * 8..i * 8 + 8].try_into().unwrap());
    }
    w
}

#[inline]
pub fn words_to_bytes(w: &[u64; CHUNK_WORDS]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for (i, wi) in w.iter().enumerate() {
        b[i * 8..i * 8 + 8].copy_from_slice(&wi.to_le_bytes());
    }
    b
}

#[inline]
fn trace_leaf(state: u64) -> [u8; 32] {
    blake(&state.to_le_bytes())
}

fn hash_pair(left: &[u8; 32], right: &[u8; 32]) -> [u8; 32] {
    let mut buf = [0u8; 64];
    buf[..32].copy_from_slice(left);
    buf[32..].copy_from_slice(right);
    blake(&buf)
}

fn le_leq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    for i in (0..32).rev() {
        if a[i] < b[i] {
            return true;
        }
        if a[i] > b[i] {
            return false;
        }
    }
    true
}

/// Canonical block seed (initial walk state) — identical to the node's `pom_block_seed`.
pub fn pom_block_seed(pre_pow_hash: &[u8; 32], timestamp: u64, nonce: u64) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"KRX-PoM-seed/v1");
    hasher.update(pre_pow_hash);
    hasher.update(&timestamp.to_le_bytes());
    hasher.update(&nonce.to_le_bytes());
    let d = hasher.finalize();
    let seed = u64::from_le_bytes(d.as_bytes()[..8].try_into().unwrap());
    seed_state(seed)
}

/// Canonical pow value — identical to the node's `pom_pow_value`.
pub fn pom_pow_value(final_state: u64, pre_pow_hash: &[u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"KRX-PoM-pow/v1");
    hasher.update(&final_state.to_le_bytes());
    hasher.update(pre_pow_hash);
    *hasher.finalize().as_bytes()
}

pub fn merkle_root(leaves: &[[u8; 32]]) -> [u8; 32] {
    assert!(!leaves.is_empty(), "merkle_root: empty leaves");
    let mut level = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        level = next;
    }
    level[0]
}

pub fn merkle_proof(leaves: &[[u8; 32]], index: usize) -> Vec<[u8; 32]> {
    let mut path = Vec::new();
    let mut level = leaves.to_vec();
    let mut idx = index;
    while level.len() > 1 {
        let sib_idx = if idx & 1 == 0 { idx + 1 } else { idx - 1 };
        let sib = if sib_idx < level.len() { level[sib_idx] } else { level[idx] };
        path.push(sib);
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        let mut i = 0;
        while i < level.len() {
            let r = if i + 1 < level.len() { level[i + 1] } else { level[i] };
            next.push(hash_pair(&level[i], &r));
            i += 2;
        }
        idx >>= 1;
        level = next;
    }
    path
}

fn verify_merkle(leaf: [u8; 32], index: u64, path: &[[u8; 32]], root: &[u8; 32]) -> bool {
    let mut acc = leaf;
    let mut idx = index;
    for sib in path {
        acc = if idx & 1 == 0 { hash_pair(&acc, sib) } else { hash_pair(sib, &acc) };
        idx >>= 1;
    }
    &acc == root
}

/// Fiat-Shamir challenge step-indices — byte-layout identical to node/pom-core.
pub fn challenges(pre_pow_hash: &[u8; 32], nonce: u64, trace_root: &[u8; 32], pow_value: &[u8; 32], t: usize, k: u32) -> Vec<u32> {
    let mut fs = [0u8; 104];
    fs[..32].copy_from_slice(pre_pow_hash);
    fs[32..40].copy_from_slice(&nonce.to_le_bytes());
    fs[40..72].copy_from_slice(trace_root);
    fs[72..104].copy_from_slice(pow_value);
    let seed = blake(&fs);
    let mut out = Vec::with_capacity(t);
    for j in 0..t as u64 {
        let mut buf = [0u8; 40];
        buf[..32].copy_from_slice(&seed);
        buf[32..].copy_from_slice(&j.to_le_bytes());
        let d = blake(&buf);
        let v = u64::from_le_bytes(d[..8].try_into().unwrap());
        out.push((v % k as u64) as u32);
    }
    out
}

/// PROVER. Re-walk the (already-won) nonce recording the trace, commit it, and open the
/// `t` FS-selected steps. `read_chunk(off)` reads the 32 B chunk at canonical chunk index
/// `off` from the resident weight blob; `weight_leaves` is the precomputed per-chunk leaf
/// set (`blake(chunk_bytes)`) over the canonical layout, used to produce weight Merkle paths.
#[allow(clippy::too_many_arguments)]
pub fn build_proof<F: Fn(u64) -> [u64; CHUNK_WORDS]>(
    tier: u8,
    pre_pow_hash: &[u8; 32],
    nonce: u64,
    seed: u64,
    n_chunks: u64,
    k: u32,
    t: usize,
    read_chunk: F,
    weight_leaves: &[[u8; 32]],
) -> PomProof {
    let mut trace = Vec::with_capacity(k as usize + 1);
    let mut state = seed;
    trace.push(state);
    let mut off = state % n_chunks;
    for _ in 0..k {
        state = transition(state, &read_chunk(off));
        trace.push(state);
        off = state % n_chunks;
    }
    let trace_leaves: Vec<[u8; 32]> = trace.iter().map(|&s| trace_leaf(s)).collect();
    let trace_root = merkle_root(&trace_leaves);
    let final_state = trace[k as usize];
    let pow_value = pom_pow_value(final_state, pre_pow_hash);

    let chs = challenges(pre_pow_hash, nonce, &trace_root, &pow_value, t, k);
    let openings = chs
        .iter()
        .map(|&i| {
            let i = i as usize;
            let sb = trace[i];
            let off = sb % n_chunks;
            PomOpening {
                state_before: sb,
                chunk: words_to_bytes(&read_chunk(off)),
                weight_path: merkle_proof(weight_leaves, off as usize),
                trace_path_before: merkle_proof(&trace_leaves, i),
                trace_path_after: merkle_proof(&trace_leaves, i + 1),
            }
        })
        .collect();

    PomProof {
        tier,
        trace_root,
        pow_value,
        final_state,
        initial_trace_path: merkle_proof(&trace_leaves, 0),
        final_trace_path: merkle_proof(&trace_leaves, k as usize),
        openings,
    }
}

/// Self-check a built proof before submit (same logic the node runs). Cheap insurance
/// against emitting a block the node will reject.
#[allow(clippy::too_many_arguments)]
pub fn verify_proof(pre_pow_hash: &[u8; 32], nonce: u64, seed: u64, proof: &PomProof, n_chunks: u64, k: u32, t: usize, r_t: &[u8; 32], target: &[u8; 32]) -> bool {
    if proof.openings.len() != t {
        return false;
    }
    if pom_pow_value(proof.final_state, pre_pow_hash) != proof.pow_value {
        return false;
    }
    if !le_leq(&proof.pow_value, target) {
        return false;
    }
    if !verify_merkle(trace_leaf(seed), 0, &proof.initial_trace_path, &proof.trace_root) {
        return false;
    }
    if !verify_merkle(trace_leaf(proof.final_state), k as u64, &proof.final_trace_path, &proof.trace_root) {
        return false;
    }
    let chs = challenges(pre_pow_hash, nonce, &proof.trace_root, &proof.pow_value, t, k);
    for (op, &i) in proof.openings.iter().zip(chs.iter()) {
        let i = i as u64;
        if !verify_merkle(trace_leaf(op.state_before), i, &op.trace_path_before, &proof.trace_root) {
            return false;
        }
        let off = op.state_before % n_chunks;
        if !verify_merkle(blake(&op.chunk), off, &op.weight_path, r_t) {
            return false;
        }
        let state_after = transition(op.state_before, &chunk_to_words(&op.chunk));
        if !verify_merkle(trace_leaf(state_after), i + 1, &op.trace_path_after, &proof.trace_root) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_chunk(off: u64) -> [u64; CHUNK_WORDS] {
        let mut c = [0u64; CHUNK_WORDS];
        for (j, w) in c.iter_mut().enumerate() {
            *w = mix64(off.wrapping_mul(CHUNK_WORDS as u64) + j as u64 + 1);
        }
        c
    }

    #[test]
    fn build_then_self_verify() {
        let (n, k, t) = (4096u64, 256u32, 32usize);
        let pph = blake(b"pph");
        let nonce = 0xabc;
        let seed = pom_block_seed(&pph, 111, nonce);
        let weight_leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        let r_t = merkle_root(&weight_leaves);

        let proof = build_proof(2, &pph, nonce, seed, n, k, t, synth_chunk, &weight_leaves);
        assert!(verify_proof(&pph, nonce, seed, &proof, n, k, t, &r_t, &[0xff; 32]));
        // borsh wire-format round-trips (same encoding the node decodes).
        let bytes = borsh::to_vec(&proof).unwrap();
        let back: PomProof = borsh::from_slice(&bytes).unwrap();
        assert!(verify_proof(&pph, nonce, seed, &back, n, k, t, &r_t, &[0xff; 32]));
        assert_eq!(back.tier, 2);
    }

    #[test]
    fn wrong_target_or_root_fails() {
        let (n, k, t) = (4096u64, 256u32, 32usize);
        let pph = blake(b"pph2");
        let nonce = 7;
        let seed = pom_block_seed(&pph, 1, nonce);
        let weight_leaves: Vec<[u8; 32]> = (0..n).map(|o| blake(&words_to_bytes(&synth_chunk(o)))).collect();
        let r_t = merkle_root(&weight_leaves);
        let proof = build_proof(0, &pph, nonce, seed, n, k, t, synth_chunk, &weight_leaves);
        assert!(!verify_proof(&pph, nonce, seed, &proof, n, k, t, &r_t, &[0u8; 32]), "zero target must fail");
        assert!(!verify_proof(&pph, nonce, seed, &proof, n, k, t, &blake(b"wrong"), &[0xff; 32]), "wrong R_T must fail");
    }
}
