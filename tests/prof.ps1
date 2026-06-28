# Build bench_pom.exe (sm_89), copy to the 4050 box, profile the PoM kernel with Nsight Compute.
# Ports p40-pearl-gemm/tests/prof.ps1. Uses the `prof` mode (one clean launch) so ncu attaches
# to exactly pom_mine.
#
# Usage:  powershell -File tests\prof.ps1
Set-Location 'E:\keryx\keryx-miner-src'
$vc   = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe'
$flags = '-O3 -std=c++17 --use_fast_math -arch=sm_89 -cudart static -Xcompiler /MT'
Remove-Item tests\bench_pom.exe -ErrorAction SilentlyContinue
cmd /c "`"$vc`" >nul 2>&1 && `"$nvcc`" $flags -o tests\bench_pom.exe tests\bench_pom.cu 2>&1"
if (-not (Test-Path tests\bench_pom.exe)) { Write-Output 'BUILD FAILED'; exit 1 }

$key = "$env:USERPROFILE\.ssh\id_rsa"
$h   = 'kfn collegiate@6.tcp.us-cal-1.ngrok.io'   # username has a literal space
$P   = 26173
scp -i $key -P $P -o StrictHostKeyChecking=accept-new -o BatchMode=yes tests\bench_pom.exe "${h}:C:/obm/bench_pom.exe" 2>&1 | Out-Null

# Same section set prof.ps1 uses — SpeedOfLight + the stall/bandwidth breakdowns that tell us
# whether the kernel is memory-, compute-, or latency-bound (the open question after the flat
# magic-modulo result on the P40).
$sec = '--section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section WarpStateStats --section SchedulerStats --section Occupancy'
ssh -i $key -p $P -o StrictHostKeyChecking=accept-new -o BatchMode=yes $h "ncu --kernel-name regex:pom_mine -c 1 $sec C:\obm\bench_pom.exe prof" 2>&1 | Tee-Object tests\ncu_pom.txt
