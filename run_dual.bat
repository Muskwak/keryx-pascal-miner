@echo off
REM ============================================================================
REM  run_dual.bat — run TWO keryx-miner instances, one per GPU, both on the
REM  --light tier (Gemma-3-4B-abliterated) post-hardfork.
REM
REM  Both GPUs mine Gemma-3-4B for now because the Qwen3-32B download from the
REM  keryx-labs.com IPFS gateway is extremely slow on this network (~0.7 MB/s).
REM  Once Qwen finishes (run\p40\ has .ok), switch P40 back to --high.
REM
REM  Instance A (P40,   CUDA ordinal 1): Gemma-3-4B-abliterated (--light)
REM  Instance B (GTX1070, CUDA ordinal 0): Gemma-3-4B-abliterated (--light)
REM
REM  GPU ordinals (from nvidia-smi, confirmed by a CUDA device query during setup):
REM     0 = NVIDIA GeForce GTX 1070  (8 GB)   -> Gemma-3-4B  (~4 GB weights)
REM     1 = Tesla P40                (24 GB)  -> Gemma-3-4B  (~4 GB weights)
REM
REM  The Gemma-3-4B model is staged at both run\p40\models\Gemma-3-4B\ and
REM  run\gpu1070\models\Gemma-3-4B\ (with .ok markers) so both skip downloads.
REM
REM  USAGE:
REM     set MINING_ADDRESS=keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3
REM     run_dual.bat
REM  Optional: set NODE_PORT=22211 for testnet (default 22110 mainnet).
REM ============================================================================

setlocal enabledelayedexpansion

if "%MINING_ADDRESS%"=="" (
  set "MINING_ADDRESS=keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3"
)

set "NODE_HOST=127.0.0.1"
set "NODE_PORT=22110"
if not "%NODE_PORT_OVERRIDE%"=="" set "NODE_PORT=%NODE_PORT_OVERRIDE%"

REM ---- CUDA 12.8 runtime DLLs on PATH (cuBLAS for the inference probe) -------
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
set "PATH=%CUDA_BIN%;%PATH%"
if not exist "%CUDA_BIN%\cublas64_12.dll" (
  echo [run_dual] WARNING: cuBLAS not in %CUDA_BIN% — the miner will refuse to start.
)

set "ROOT=%~dp0"
set "CUDA_BIN_FLAG=--cuda-no-blocking-sync"

echo [run_dual] Starting dual-instance mining (post-hardfork abliterated lineup):
echo [run_dual]   P40   (CUDA 1): Qwen3-32B-abliterated  --high   ^(downloads ~19.5 GB^)
echo [run_dual]   1070  (CUDA 0): Gemma-3-4B-abliterated  --light  ^(staged^)
echo [run_dual]   node: grpc://%NODE_HOST%:%NODE_PORT%   reward: %MINING_ADDRESS%
echo.

REM ============================================================================
REM Instance A — Tesla P40 (ordinal 1) → Gemma-3-4B-abliterated (--light)
REM Running --light until the Qwen3-32B download completes (currently stalled at
REM ~4.7 GB / 19.8 GB on the keryx-labs.com IPFS gateway — switch back to --high
REM once run\p40\models\Qwen3-32B\.ok exists).
REM Gemma is already staged at run\p40\models\Gemma-3-4B\ so no download needed.
REM KERYX_CUDA_DEVICE=1 reads the P40's line for the VRAM/power gate.
REM --no-legacy skips pre-hardfork model prefetch.
REM ============================================================================
echo [run_dual] Launching P40 instance (Gemma-3-4B) in new window…
start "Keryx P40 / Gemma-3-4B" cmd /c ^
  "set PATH=%CUDA_BIN%;%PATH% && set KERYX_CUDA_DEVICE=1 && cd /d %ROOT%run\p40 && ^
   keryx-miner.exe -a %MINING_ADDRESS% -s %NODE_HOST% --port %NODE_PORT% ^
     --light --no-legacy --cuda-device 1 %CUDA_BIN_FLAG% && pause"

REM ============================================================================
REM Instance B — GTX 1070 (ordinal 0) → Gemma-3-4B-abliterated
REM Model already staged, so this instance skips download and mines immediately
REM (once the node is up post-hardfork).
REM KERYX_CUDA_DEVICE=0 reads the 1070's line (ordinal 0 is correct here).
REM --no-legacy skips the pre-hardfork tinyllama prefetch (~2 GB saved); only Gemma-3-4B (staged)
REM is announced. Remove --no-legacy here if you want to mine the pre-fork window on the 1070.
REM ============================================================================
echo [run_dual] Launching GTX 1070 instance (Gemma-3-4B) in new window…
start "Keryx 1070 / Gemma-3-4B" cmd /c ^
  "set PATH=%CUDA_BIN%;%PATH% && set KERYX_CUDA_DEVICE=0 && cd /d %ROOT%run\gpu1070 && ^
   keryx-miner.exe -a %MINING_ADDRESS% -s %NODE_HOST% --port %NODE_PORT% ^
      --light --no-legacy --cuda-device 0 %CUDA_BIN_FLAG% && pause"

echo.
echo [run_dual] Both instances launched in separate windows.
echo [run_dual]   - Both are mining Gemma-3-4B-abliterated (--light) since Qwen3-32B
echo [run_dual]     download stalled at ~4.7/19.8 GB on the slow IPFS gateway.
echo [run_dual]   - Each window should show "Model files ready" quickly (both staged).
echo [run_dual]   - Once your node is up: look for "GPU inference verified — cuBLAS loaded"
echo [run_dual]     and a hashrate line in each window.
echo [run_dual]   - escrow.key in each work dir holds your OPoI reward key — back both up.
echo [run_dual]   - TO SWITCH P40 BACK TO --HIGH: kill its window, update the command
echo [run_dual]     above to --high, and re-run once Qwen3-32B has .ok.
exit /b 0
