// Keryx Proof-of-Model mining kernel — TESLA P40 / sm_61 (Pascal GP102) TUNED.
//
// DROP-IN replacement for the upstream cuda/pom_mine.cu: identical `extern "C" pom_mine`
// signature and arg order, so src/pom_gpu.rs loads this PTX unchanged. Only the device-side
// memory access is optimized for Pascal.
//
// Device primitives (mix64 / pom_seed_fold / pom_pow_fold / pom_le_leq) are byte-identical to
// the original kernel and to the host pom.rs — the node re-derives the same values, so the
// math MUST match. Verified against the host via the upstream self-check + R_T pin.
//
// P40/sm_61-specific optimizations vs the generic upstream kernel:
//   1. __ldg() on weight gathers  — routes random reads through Pascal's read-only texture
//      cache. sm_61 has NO L1 data cache for globals (loads go straight to L2), so __ldg is
//      the single biggest latency win for the data-dependent walk.
//   2. Cooperative shared-memory load of `prefix[]` — the per-step binary search hits smem
//      (~20 cyc) instead of 256× global reads (~400 cyc) per nonce. T (tensor count) fits smem.
//   3. 256 threads/block keeps the binary-search loop's register pressure within sm_61's 64K
//      registers/SM for high warps-in-flight, hiding the ~400-cycle global gather latency.
//
// The signature/arg order matches upstream exactly so pom_gpu.rs::mine() pushes args unchanged:
//   bases, prefix, T, n_total_chunks, K, p0,p1,p2,p3, time_, t0,t1,t2,t3, start, batch, winner

#include <cstdint>

extern "C" {

__device__ __forceinline__ unsigned long long mix64(unsigned long long x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

__device__ __forceinline__ unsigned long long pom_seed_fold(
    unsigned long long nonce, unsigned long long time_,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3) {
    unsigned long long s = mix64(nonce ^ 0x4B65727978531ULL);
    s = mix64(s ^ time_);
    s = mix64(s ^ p0); s = mix64(s ^ p1); s = mix64(s ^ p2); s = mix64(s ^ p3);
    return s;
}

__device__ __forceinline__ void pom_pow_fold(
    unsigned long long fin, unsigned long long p0, unsigned long long p1,
    unsigned long long p2, unsigned long long p3, unsigned long long out[4]) {
    out[0] = mix64(fin ^ p0 ^ 0x9E3779B97F4A7C15ULL);
    out[1] = mix64(out[0] ^ p1 ^ 0xC2B2AE3D27D4EB4FULL);
    out[2] = mix64(out[1] ^ p2 ^ 0x165667B19E3779F9ULL);
    out[3] = mix64(out[2] ^ p3 ^ 0xD6E8FEB86659FD93ULL);
}

__device__ __forceinline__ bool pom_le_leq(const unsigned long long a[4],
                                           unsigned long long b0, unsigned long long b1,
                                           unsigned long long b2, unsigned long long b3) {
    if (a[3] != b3) return a[3] < b3;
    if (a[2] != b2) return a[2] < b2;
    if (a[1] != b1) return a[1] < b1;
    return a[0] <= b0;
}

// Find the tensor index whose [prefix[t], prefix[t+1]) chunk range contains `off`, using the
// shared-memory copy of `prefix`. Returns lo such that prefix[lo] <= off < prefix[lo+1].
__device__ __forceinline__ unsigned int find_tensor(const unsigned long long* __restrict__ sprefix,
                                                    unsigned int T, unsigned long long off) {
    unsigned int lo = 0, hi = T;
    while (lo + 1 < hi) {
        unsigned int mid = (lo + hi) >> 1;
        if (sprefix[mid] <= off) lo = mid; else hi = mid;
    }
    return lo;
}

__global__ void pom_mine(
    const unsigned long long* __restrict__ bases,
    const unsigned long long* __restrict__ prefix,
    unsigned int T,
    unsigned long long n_total_chunks, unsigned int K,
    unsigned long long p0, unsigned long long p1, unsigned long long p2, unsigned long long p3,
    unsigned long long time_,
    unsigned long long t0, unsigned long long t1, unsigned long long t2, unsigned long long t3,
    unsigned long long nonce_base, unsigned long long n_nonces,
    unsigned long long* winner) {

    extern __shared__ unsigned long long sprefix[];  // shared copy of prefix[0..T]

    // Cooperative load of the offset table into shared memory — every thread in the block
    // touches the table ~256× per nonce; smem latency (~20 cyc) beats global (~400 cyc) hard.
    unsigned int tid = threadIdx.x;
    for (unsigned int i = tid; i <= T; i += blockDim.x) {
        sprefix[i] = prefix[i];
    }
    __syncthreads();

    unsigned long long gid = (unsigned long long)blockIdx.x * blockDim.x + tid;
    if (gid >= n_nonces) return;
    unsigned long long nonce = nonce_base + gid;

    unsigned long long state = pom_seed_fold(nonce, time_, p0, p1, p2, p3);
    unsigned long long off = state % n_total_chunks;
    for (unsigned int i = 0; i < K; i++) {
        unsigned int lo = find_tensor(sprefix, T, off);
        unsigned long long local = off - sprefix[lo];
        // bases[lo] is a device pointer to tensor lo's u64 data. __ldg routes the random
        // 4-u64 gather through Pascal's read-only data cache — the dominant P40 win.
        const unsigned long long* p =
            (const unsigned long long*)__ldg(bases + lo);
        unsigned long long base = local * 4ULL;
        unsigned long long h = state;
        h ^= __ldg(p + base);
        h ^= __ldg(p + base + 1);
        h ^= __ldg(p + base + 2);
        h ^= __ldg(p + base + 3);
        state = mix64(h);
        off = state % n_total_chunks;
    }
    unsigned long long pv[4];
    pom_pow_fold(state, p0, p1, p2, p3, pv);
    if (pom_le_leq(pv, t0, t1, t2, t3)) {
        // Min atomically so the lowest nonce wins (deterministic on ties).
        atomicMin(winner, nonce);
    }
}

} // extern "C"
