@echo off
REM ============================================================================
REM  run_p40.bat — launch the keryx-miner (P40-tuned) against your local node.
REM
REM  Sets up everything the binary needs at runtime:
REM    - CUDA 12.8 runtime DLLs on PATH (cudart, cublas, cublasLt — the inference
REM      probe at main.rs:513 refuses to start without cuBLAS).
REM    - protoc already used at build time (not needed at runtime).
REM
REM  The Pascal-tuned PoM kernel (cuda/pom_mine.cu with __ldg + shared-mem prefix)
REM  is compiled into the binary — nothing to set here for that.
REM
REM  The Gemma-3-4B model for --light is staged at:
REM    target\release\models\Gemma-3-4B\model.gguf  (with a .ok marker)
REM  so the miner skips the IPFS download at boot.
REM
REM  USAGE:
REM    set MINING_ADDRESS=keryx:qz...   (your reward address — REQUIRED)
REM    run_p40.bat                       (defaults below; edit NODE/PORT/NETWORK as needed)
REM
REM  When your node is back up post-hardfork, just run this. The first successful
REM  output line is "GPU inference verified — cuBLAS loaded successfully."
REM ============================================================================

setlocal enabledelayedexpansion

REM ---- Required: your Keryx reward address ------------------------------------
if "%MINING_ADDRESS%"=="" (
  echo [run_p40] ERROR: set the MINING_ADDRESS env var first, e.g.:
  echo   set MINING_ADDRESS=keryx:qzrX8JbHk...your...address
  echo.
  echo [run_p40] Aborting — a mining address is mandatory (the coinbase pays here).
  exit /b 2
)

REM ---- Node connection (edit to match your keryxd) ----------------------------
REM  Solo mining against your own node uses grpc://. For a pool, set NODE to the
REM  pool URL with the stratum+tcp:// schema instead (e.g. stratum+tcp://pool.example:3333).
set "NODE_HOST=127.0.0.1"
set "NODE_PORT=22110"
REM  Network: mainnet=22110, testnet=22211. Override via NODE_PORT if needed.

REM ---- CUDA runtime DLLs on PATH (cuBLAS is what the probe checks for) --------
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
set "PATH=%CUDA_BIN%;%PATH%"
if not exist "%CUDA_BIN%\cublas64_12.dll" (
  echo [run_p40] WARNING: cuBLAS not found in %CUDA_BIN%.
  echo [run_p40]   The miner will refuse to start (OPoI inference probe fails without it).
  echo [run_p40]   Install CUDA 12.x runtime or fix CUDA_BIN above.
  echo.
)

REM ---- cuDNN note (not fatal, but some inference ops need it) -----------------
REM  candle uses cuDNN for conv/some norm ops. It is NOT required to start mining
REM  (the probe only checks cuBLAS), but missing it can cause mid-run inference
REM  failures on certain challenges. If you hit "cudnn" load errors, install
REM  cuDNN 8/9 for CUDA 12 and add its bin to PATH.

REM ---- GPU selection (default: auto-pick all CUDA devices) --------------------
REM  Your rig has Tesla P40 (device 0) + GTX 1070 (device 1). To mine ONLY on the
REM  P40, set CUDA_DEVICE=0. To use both, leave blank.
if not "%CUDA_DEVICE%"=="" (
  set "DEV_FLAG=--cuda-device %CUDA_DEVICE%"
) else (
  set "DEV_FLAG="
)

REM ---- Tier: --light = Gemma-3-4B (fits P40 24GB with headroom) ----------------
REM  Other tiers: (none)=Dolphin-8B, --high=Qwen3-32B(19.5GB,tight on P40),
REM  --very-high=Llama-3.3-70B(needs 48GB+ — not for the P40).
set "TIER_FLAG=--light"

REM ---- Launch ----------------------------------------------------------------
echo [run_p40] Starting keryx-miner v0.3.3 (P40-tuned kernel)…
echo [run_p40]   address : %MINING_ADDRESS%
echo [run_p40]   node    : grpc://%NODE_HOST%:%NODE_PORT%
echo [run_p40]   tier    : Gemma-3-4B (--light)
echo [run_p40]   devices : %CUDA_DEVICE%(auto if blank)
echo.

REM  Run from target\release so the binary finds models\Gemma-3-4B\ relative to itself.
pushd "%~dp0target\release"

keryx-miner.exe ^
  -a "%MINING_ADDRESS%" ^
  -s "%NODE_HOST%" ^
  --port %NODE_PORT% ^
  %TIER_FLAG% ^
  %DEV_FLAG%

set RC=%ERRORLEVEL%
popd

echo.
echo [run_p40] keryx-miner exited rc=%RC%
exit /b %RC%
