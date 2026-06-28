// bench_pom.cu — standalone benchmark + Nsight profile harness for the keryx PoM walk kernel.
//
// Ports the p40-pearl-gemm/tests/bench_ampere.cu pattern to keryx-miner: the PoM kernel
// (cuda/pom_mine.cu) is self-contained (no candle/node), so we #include it and drive it with
// SYNTHETIC weights — same structure the host builds (per-tensor bases[] + canonical prefix[]
// offset table) but filled deterministically, so no 2.5GB GGUF needs to ship to the box.
//
// Build (from keryx-miner-src/):
//   nvcc -O3 -std=c++17 --use_fast_math -arch=sm_89 -cudart static -Xcompiler /MT \
//        -o tests\bench_pom.exe tests\bench_pom.cu
// Run:
//   bench_pom.exe [mode]        modes: bench | prof | minepipe  (default bench)
//
//   bench    : bit-exact CPU-vs-GPU check + steady-state MH/s at the real batch size
//   prof     : ONE clean launch for `ncu -k pom_mine` profiling
//   minepipe : replicate the miner's per-batch sync cadence (host wall-clock timed) to isolate
//              GPU idle during the host round-trip — tests the "launch-overhead bound" hypothesis.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <chrono>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), err); \
        exit(1); \
    } \
} while(0)

// ---- The PoM kernel under test (byte-identical to what the miner ships) ----------------
// pom_mine.cu defines `extern "C" __global__ pom_mine`. We bring it in so a single nvcc
// invocation compiles kernel + bench together at the target arch.
#include "../cuda/pom_mine.cu"

// ---- Constants mirroring src/pom.rs + src/miner.rs ------------------------------------
static constexpr uint32_t K = 256;            // POM_WALK_STEPS
static constexpr uint64_t POM_BATCH = 1ull << 21;   // src/miner.rs POM_BATCH (2M nonces/launch)
static constexpr size_t   CHUNK_BYTES = 32;   // src/pom_gpu.rs CHUNK_BYTES (4 × u64)
static constexpr uint32_t THREADS = 128;      // pom_gpu.rs block_dim

// ---- mix64 + host walk (byte-identical to pom.rs) for the bit-exact check -------------
static uint64_t mix64h(uint64_t x) {
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
// Portable high half of a 64x64->128 unsigned multiply (32-bit halves — MSVC has no __int128).
// Used by both the magic-modulo verifier and host_fast_mod, so it lives up here with the walk.
static uint64_t mulhi64h(uint64_t a, uint64_t b) {
    uint64_t aL = (uint32_t)a, aH = a >> 32;
    uint64_t bL = (uint32_t)b, bH = b >> 32;
    uint64_t lo  = aL * bL;
    uint64_t mid = aH * bL + (lo >> 32);
    uint64_t hi  = aH * bH + (mid >> 32);
    mid = (uint32_t)mid + aL * bH;
    return hi + (mid >> 32);
}
static uint64_t seed_fold_h(uint64_t nonce, uint64_t t, uint64_t p[4]) {
    uint64_t s = mix64h(nonce ^ 0x4B65727978531ULL);
    s = mix64h(s ^ t);
    s = mix64h(s ^ p[0]); s = mix64h(s ^ p[1]); s = mix64h(s ^ p[2]); s = mix64h(s ^ p[3]);
    return s;
}
// Full host-side PoM walk for one nonce. Mirrors cuda/pom_mine.cu exactly: seed-fold, K steps
// of (find_tensor via prefix[] binary search, 4-u64 gather from the tensor's host buffer, XOR
// fold, mix64, fast_mod advance), then pow_fold. Returns the 4-word proof value in out[4].
// The gather reads from the HOST copies of the tensor data — we keep those around specifically
// so the CPU walk can re-derive what the GPU computed and we can prove the kernel is bit-exact.
static void pow_fold_h(uint64_t fin, uint64_t p[4], uint64_t out[4]) {
    out[0] = mix64h(fin ^ p[0] ^ 0x9E3779B97F4A7C15ULL);
    out[1] = mix64h(out[0] ^ p[1] ^ 0xC2B2AE3D27D4EB4FULL);
    out[2] = mix64h(out[1] ^ p[2] ^ 0x165667B19E3779F9ULL);
    out[3] = mix64h(out[2] ^ p[3] ^ 0xD6E8FEB86659FD93ULL);
}
static bool le_leq_h(const uint64_t a[4], uint64_t b[4]) {
    if (a[3] != b[3]) return a[3] < b[3];
    if (a[2] != b[2]) return a[2] < b[2];
    if (a[1] != b[1]) return a[1] < b[1];
    return a[0] <= b[0];
}
// host_fast_mod mirrors the kernel's fast_mod bit-for-bit (same mulhi + shift), using the
// portable 32-bit-halves mulhi (no __int128 on MSVC). magic==0 → plain %.
static uint64_t host_fast_mod(uint64_t a, uint64_t magic, uint32_t shift, uint64_t d) {
    if (magic == 0) return a % d;
    uint64_t q = mulhi64h(a, magic) >> shift;
    return a - q * d;
}

// ---- Magic-number modulo for the bench -----------------------------------------------
// The bench's job is to MEASURE the kernel, not re-derive the magic — the live miner already
// computes + self-tests it (src/pom_gpu.rs), and the node re-walks winning nonces with native %.
// So we use the proven constant for the production n_chunks (77809197 = Gemma-3-4B @ 32B/chunk),
// validated in Python (magic=0xdccb7a9e740e52c0, shift=26, 0 failures over 1M+ inputs), and
// confirm it on this host via the portable mulhi (mulhi64h, defined above with the host walk).
struct ModMagic { uint64_t magic; uint32_t shift; };
// Proven for n_chunks=77809197 (see tests/magic_verify.py). Re-checked here at startup.
static ModMagic mod_magic_for_chunks(uint64_t n_chunks) {
    if (n_chunks == 77809197ull) return {0xdccb7a9e740e52c0ull, 26};
    return {0, 0};  // unknown → sentinel, kernel uses plain %
}
// Confirm the constant actually produces bit-exact % on this machine before trusting it.
static bool verify_mod(const ModMagic& m, uint64_t d) {
    if (m.magic == 0 || m.shift > 63) return false;
    auto check = [&](uint64_t a) {
        uint64_t q = mulhi64h(a, m.magic) >> m.shift;
        return a - q * d == a % d;
    };
    uint64_t rng = 0;
    for (int i = 0; i < 200000; i++) {
        rng += 0x9E3779B97F4A7C15ULL;
        uint64_t z = rng;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        if (!check(z)) return false;
    }
    for (uint64_t z : {0ull,1ull,d-1,d,d+1,~0ull,d*2-1,d*2,d*2+1,d*3})
        if (!check(z)) return false;
    return true;
}

// ---- Build the gather index from synthetic per-tensor weights -------------------------
// Mirrors PomGpuMiner::load(): names-sorted tensors → bases[lo] (device ptr) + prefix[] offset
// table. We synthesize T tensors of varying size so the binary-search walk hits realistic
// branch patterns. All up to N_TOTAL chunks live in device memory (like the resident weights).
#include <vector>
#include <algorithm>
#include <array>
struct Gather {
    uint64_t* bases = nullptr;    // device: T entries, each a device ptr to that tensor's u64 chunks
    uint64_t* prefix = nullptr;   // device: T+1 entries, canonical chunk offsets
    uint32_t  T = 0;
    uint64_t  n_chunks = 0;
    std::vector<uint64_t*> tensor_dev;  // owns the per-tensor device buffers
    std::vector<uint64_t> prefix_host;  // host mirror of prefix (for the CPU walk / bit-exact check)
    // Host mirrors of each tensor's u64 chunk data, kept so the CPU reference walk can read the
    // exact same bytes the GPU gathers — the bit-exact check XOR-folds these to compare against
    // the kernel's winner. Indexed tensor_host[lo] points at sz*4 contiguous u64s.
    std::vector<std::vector<uint64_t>> tensor_host;
    std::vector<uint64_t> tensor_sz;     // chunk count per tensor (= tensor_host[lo].size()/4)
};
static Gather build_gather(uint64_t target_chunks, uint32_t target_tensors) {
    // Synthesize `target_tensors` tensors (Gemma-3-4B has 444) sized to total `target_chunks`,
    // so the prefix[] shared-mem table (T+1 u64s) stays small (~3.5 KB — well under the smem
    // cap). A handful of tensors get most of the weight (real LLMs are dominated by a few big
    // attention/FFN matrices), the rest are small (norms, embeddings tail).
    Gather g;
    g.prefix_host = {0};
    uint64_t made = 0;
    uint64_t seed = 0x1234;
    for (uint32_t t = 0; t < target_tensors && made < target_chunks; t++) {
        seed = mix64h(seed);
        uint64_t sz;
        if (t < target_tensors / 4) {
            // big tensors: each ~30-50% of remaining, so a few dominate (like real FFN weights)
            uint64_t remaining = target_chunks - made;
            sz = remaining / (4 + (seed % 4));
        } else {
            // small tensors: 1..1024 chunks (norms, biases)
            sz = 1 + (seed % 1024);
        }
        if (made + sz > target_chunks) sz = target_chunks - made;
        size_t bytes = (size_t)sz * CHUNK_BYTES;
        uint64_t* dev;
        CUDA_CHECK(cudaMalloc(&dev, bytes));
        // deterministic fill so any future bit-exact check is reproducible
        std::vector<uint64_t> fill(sz * 4);
        for (uint64_t i = 0; i < sz * 4; i++)
            fill[i] = mix64h(i * 0x9E3779B97F4A7C15ULL + g.T);
        CUDA_CHECK(cudaMemcpy(dev, fill.data(), bytes, cudaMemcpyHostToDevice));
        g.tensor_dev.push_back(dev);
        g.tensor_host.push_back(std::move(fill));   // keep host copy for the CPU reference walk
        g.tensor_sz.push_back(sz);
        made += sz;
        g.prefix_host.push_back(made);
        g.T++;
    }
    g.n_chunks = made;
    std::vector<uint64_t> bases_host(g.T);
    for (uint32_t i = 0; i < g.T; i++) bases_host[i] = (uint64_t)g.tensor_dev[i];
    CUDA_CHECK(cudaMalloc(&g.bases, (size_t)g.T * 8));
    CUDA_CHECK(cudaMemcpy(g.bases, bases_host.data(), (size_t)g.T * 8, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&g.prefix, (size_t)(g.T + 1) * 8));
    CUDA_CHECK(cudaMemcpy(g.prefix, g.prefix_host.data(), (size_t)(g.T + 1) * 8, cudaMemcpyHostToDevice));
    return g;
}

// ---- Launch the kernel exactly like pom_gpu.rs::mine() --------------------------------
static double launch(const Gather& g, const ModMagic& mm, uint64_t p[4], uint64_t tg[4],
                     uint64_t timestamp, uint64_t nonce_base, uint64_t batch, uint64_t* winner_dev) {
    CUDA_CHECK(cudaMemset(winner_dev, 0xff, 8));
    uint32_t grid = (uint32_t)((batch + THREADS - 1) / THREADS);
    size_t smem = (size_t)(g.T + 1) * 8;
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    pom_mine<<<grid, THREADS, smem>>>(
        g.bases, g.prefix, g.T, g.n_chunks, K, mm.magic, mm.shift,
        p[0], p[1], p[2], p[3], timestamp,
        tg[0], tg[1], tg[2], tg[3],
        nonce_base, batch, winner_dev);
    // Check the launch itself (not just the surrounding API calls) — a smem-too-large or
    // invalid-grid launch fails asynchronously and would otherwise make the kernel look
    // absurdly fast (the prior bug: T=37836 → 295KB smem > cap → instant no-op launch).
    CUDA_CHECK(cudaGetLastError());
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms;
}

// ---- CPU reference walk (byte-identical to the kernel) for the bit-exact check ----------
// Walks one nonce the same way pom_mine does, reading from the HOST tensor mirrors. Returns the
// 4-word proof value in out[4]. If the GPU gather (vectorized or not) ever diverges from this,
// the check mode catches it: a winning nonce on the GPU must produce le_leq(here, target).
static void host_walk(const Gather& g, const ModMagic& mm, uint64_t p[4],
                      uint64_t timestamp, uint64_t nonce, uint64_t out[4]) {
    uint64_t state = seed_fold_h(nonce, timestamp, p);
    uint64_t off = host_fast_mod(state, mm.magic, mm.shift, g.n_chunks);
    // find_tensor via prefix_host[] (same binary search as the kernel's find_tensor)
    auto& pf = g.prefix_host;
    for (uint32_t i = 0; i < K; i++) {
        uint32_t lo = 0, hi = g.T;
        while (lo + 1 < hi) {
            uint32_t mid = (lo + hi) >> 1;
            if (pf[mid] <= off) lo = mid; else hi = mid;
        }
        uint64_t local = off - pf[lo];
        const uint64_t* tp = g.tensor_host[lo].data();
        uint64_t base = local * 4ULL;
        uint64_t h = state;
        h ^= tp[base];
        h ^= tp[base + 1];
        h ^= tp[base + 2];
        h ^= tp[base + 3];
        state = mix64h(h);
        off = host_fast_mod(state, mm.magic, mm.shift, g.n_chunks);
    }
    pow_fold_h(state, p, out);
}

int main(int argc, char** argv) {
    const char* mode = (argc >= 2) ? argv[1] : "bench";
    // Optional device index as argv[2] (after mode) — defaults to 0. Lets us A/B on a multi-GPU
    // box (e.g. --device 1 to target the P40 when the 1070 is device 0) without fighting
    // CUDA_VISIBLE_DEVICES ordering quirks across driver/CUDA versions.
    int dev = 0;
    for (int i = 2; i < argc; i++) {            // scan all args past mode for --device N
        if (!strcmp(argv[i], "--device") && i + 1 < argc) dev = atoi(argv[++i]);
    }
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("Device: %s (sm_%d%d, %d SMs, L2=%d KB)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount,
           (int)(prop.l2CacheSize / 1024));

    // Realistic chunk count for Gemma-3-4B at 32B/chunk: ~77.8M (2.49GB / 32).
    uint64_t target_chunks = 77809197;
    printf("Building synthetic gather (N=%llu chunks, ~%.1f GB)...\n",
           (unsigned long long)target_chunks, target_chunks * 32.0 / 1e9);
    // 444 tensors = the real Gemma-3-4B GGUF tensor count (verified from its header). Keeps the
    // prefix[] shared-mem table small (~3.5 KB) so the launch is valid and the kernel really runs.
    Gather g = build_gather(target_chunks, 444);
    printf("  T=%u tensors, n_chunks=%llu\n", g.T, (unsigned long long)g.n_chunks);

    ModMagic mm = mod_magic_for_chunks(g.n_chunks);
    bool ok = verify_mod(mm, g.n_chunks);
    printf("  magic modulo: magic=0x%llx shift=%u  (%s)\n",
           (unsigned long long)mm.magic, mm.shift,
           ok ? "ACTIVE (fast path, verified)" : (mm.magic ? "VERIFY FAILED -> plain %" : "sentinel (plain %)"));
    if (!ok) mm = {0, 0};

    uint64_t p[4]  = {0x1111111111111111ull, 0x2222222222222222ull,
                      0x3333333333333333ull, 0x4444444444444444ull};
    uint64_t tg[4] = {0,0,0,0};   // target=0 → no winner, clean timing
    uint64_t timestamp = 0xABCD;
    uint64_t* winner_dev; CUDA_CHECK(cudaMalloc(&winner_dev, 8));

    // ---------- check: bit-exact GPU-vs-CPU walk over a small batch ----------
    // The correctness gate for any kernel change (vectorized gather, magic modulo, etc.).
    // Strategy: pick a TIGHT target so only a handful of nonces win, then brute-force the CPU
    // reference walk over the whole batch to find the true min winning nonce and its proof. The
    // GPU's atomicMin(winner) must return that exact nonce — and host_walk on it must re-derive
    // a proof that is genuinely ≤ target (proving the proof value, not just the win decision,
    // matches). This is the same property the node relies on when it re-walks winners with native %.
    if (strcmp(mode, "check") == 0) {
        const uint64_t CBATCH = 1ull << 15;          // 32k nonces — enough to exercise the gather
        // First pass: CPU-walk the whole batch to find a target that yields a SMALL number of
        // winners (so the min-winner test is discriminating, not "everything wins"). Pick the
        // median proof as the target → ~half the batch qualifies, but we then TIGHTEN to the
        // minimum proof seen so only the tightest nonces win.
        std::vector<std::array<uint64_t,4>> proofs(CBATCH);
        for (uint64_t n = 0; n < CBATCH; n++)
            host_walk(g, mm, p, timestamp, n, proofs[n].data());
        // Find the minimum proof (lexicographic on the big-end [3,2,1,0] order le_leq_h uses).
        uint64_t best = 0;
        for (uint64_t n = 1; n < CBATCH; n++)
            if (le_leq_h(proofs[n].data(), proofs[best].data())) best = n;
        // Set target = that minimum proof. Now exactly the nonces whose proof == min qualify; the
        // GPU must return the lowest such nonce (atomicMin picks the min nonce on ties).
        uint64_t tight[4] = {proofs[best][0], proofs[best][1], proofs[best][2], proofs[best][3]};
        uint64_t ref_min = ~0ull;
        for (uint64_t n = 0; n < CBATCH; n++)
            if (le_leq_h(proofs[n].data(), tight)) { ref_min = n; break; }

        launch(g, mm, p, tight, timestamp, 0, CBATCH, winner_dev);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint64_t gpu_winner = ~0ull;
        CUDA_CHECK(cudaMemcpy(&gpu_winner, winner_dev, 8, cudaMemcpyDeviceToHost));

        // Re-derive the winner's proof on the CPU and confirm it's genuinely ≤ target.
        uint64_t hw[4]; host_walk(g, mm, p, timestamp, gpu_winner, hw);
        bool proof_valid = le_leq_h(hw, tight);

        printf("\ncheck: batch=%llu, target = min proof in batch (tight)\n", (unsigned long long)CBATCH);
        printf("  CPU min winning nonce : %llu\n", (unsigned long long)ref_min);
        printf("  GPU min winning nonce : %llu\n", (unsigned long long)gpu_winner);
        printf("  host_walk(gpu_winner) <= target : %s\n", proof_valid ? "YES" : "NO");
        if (gpu_winner == ref_min && proof_valid) {
            printf("  RESULT: PASS — GPU and CPU agree on min winning nonce; proof bit-exact\n");
        } else {
            printf("  RESULT: FAIL — kernel diverged from CPU reference walk!\n");
            return 1;
        }
    }

    // ---------- bit-exact check: GPU walk vs CPU walk for a few nonces ----------
    if (strcmp(mode, "bench") == 0) {
        // ... correctness check + steady-state MH/s at POM_BATCH
        // warmup (ramp clocks)
        for (int i = 0; i < 20; i++) launch(g, mm, p, tg, timestamp, i * POM_BATCH, POM_BATCH, winner_dev);
        CUDA_CHECK(cudaDeviceSynchronize());
        int iters = 50;
        double total = 0;
        for (int i = 0; i < iters; i++)
            total += launch(g, mm, p, tg, timestamp, i * POM_BATCH, POM_BATCH, winner_dev);
        double ms = total / iters;
        double mhs = (double)POM_BATCH / (ms / 1000.0) / 1e6;
        printf("\nbench: %.3f ms/batch -> %.3f MH/s  (batch=%llu, iters=%d)\n",
               ms, mhs, (unsigned long long)POM_BATCH, iters);
    }

    // ---------- prof: ONE clean launch for ncu ----------
    if (strcmp(mode, "prof") == 0) {
        // warmup so the one profiled launch is steady-state
        for (int i = 0; i < 5; i++) launch(g, mm, p, tg, timestamp, i * POM_BATCH, POM_BATCH, winner_dev);
        CUDA_CHECK(cudaDeviceSynchronize());
        launch(g, mm, p, tg, timestamp, 0, POM_BATCH, winner_dev);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("prof: launched one batch (%llu nonces) for ncu\n", (unsigned long long)POM_BATCH);
    }

    // ---------- minepipe: miner's per-batch sync cadence, host wall-clock ----------
    // This is the KEY experiment: if real MH/s < bench MH/s, the gap is host round-trip.
    if (strcmp(mode, "minepipe") == 0) {
        int batches = (argc >= 3 ? atoi(argv[2]) : 500);
        // warmup
        for (int i = 0; i < 20; i++) launch(g, mm, p, tg, timestamp, i * POM_BATCH, POM_BATCH, winner_dev);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        uint64_t hwinner = 0;
        for (int i = 0; i < batches; i++) {
            launch(g, mm, p, tg, timestamp, i * POM_BATCH, POM_BATCH, winner_dev);
            CUDA_CHECK(cudaDeviceSynchronize());                 // the miner's per-batch sync
            CUDA_CHECK(cudaMemcpy(&hwinner, winner_dev, 8, cudaMemcpyDeviceToHost));  // D2H read
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / batches;
        double mhs = (double)POM_BATCH / (ms / 1000.0) / 1e6;
        printf("\nminepipe: %.3f ms/batch (incl sync+D2H) -> %.3f MH/s  (batches=%d)\n",
               ms, mhs, batches);
        printf("  compare to `bench` mode — the gap is host/launch overhead the kernel can't hide\n");
    }

    cudaFree(winner_dev);
    for (auto* d : g.tensor_dev) cudaFree(d);
    cudaFree(g.bases); cudaFree(g.prefix);
    return 0;
}
