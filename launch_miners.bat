@echo off
set "CUDA_BIN=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin"
set "MINING_ADDRESS=keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3"

echo Launching P40 (GPU 1, --light)...
start "Keryx P40" cmd /c ^
  "set CUDA_VISIBLE_DEVICES=1 && set PATH=%CUDA_BIN%;%%PATH%% && set KERYX_CUDA_DEVICE=0 && cd /d %~dp0run\p40 && keryx-miner.exe -a %MINING_ADDRESS% -s grpc://127.0.0.1:22110 --light --no-legacy && pause"

echo Launching 1070 (GPU 0, --light)...
start "Keryx 1070" cmd /c ^
  "set CUDA_VISIBLE_DEVICES=0 && set PATH=%CUDA_BIN%;%%PATH%% && set KERYX_CUDA_DEVICE=0 && cd /d %~dp0run\gpu1070 && keryx-miner.exe -a %MINING_ADDRESS% -s grpc://127.0.0.1:22110 --light --no-legacy && pause"

echo Both miners launched.
pause
