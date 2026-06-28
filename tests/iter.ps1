# Build bench_pom.exe (sm_89), copy to the 4050 box, run it.
# Ports p40-pearl-gemm/tests/iter.ps1 to keryx-miner. The PoM kernel (cuda/pom_mine.cu) is
# self-contained, so the bench #includes it directly — no candle/node needed on the box, and
# weights are synthetic (no 2.5GB GGUF to ship).
#
# Usage:  powershell -File tests\iter.ps1 [mode [args...]]
#   modes: bench | prof | minepipe   (default bench)
# RUNCMD: run an arbitrary remote command instead (probes, ncu, etc.)
#         e.g.  $env:RUNCMD='hostname'; powershell -File tests\iter.ps1
# PUSHSRC/PUSHDST: scp a local file to the box.
Set-Location 'E:\keryx\keryx-miner-src'
$key = "$env:USERPROFILE\.ssh\id_rsa"
# Username is literally "kfn collegiate" (with the space) — a Windows local account.
$h   = 'kfn collegiate@6.tcp.us-cal-1.ngrok.io'
$P   = 26173

if ($env:PUSHSRC) {
    scp -i $key -P $P -o StrictHostKeyChecking=accept-new -o BatchMode=yes $env:PUSHSRC "${h}:$env:PUSHDST"
}
if ($env:RUNCMD) {
    ssh -i $key -p $P -o StrictHostKeyChecking=accept-new -o BatchMode=yes $h $env:RUNCMD
    exit $LASTEXITCODE
}

$vc   = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
$nvcc = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe'
# sm_89 = the 4050 box (Ada). Mirror of prof.ps1's flags, minus the cutlass includes (PoM needs none).
$flags = '-O3 -std=c++17 --use_fast_math -arch=sm_89 -cudart static -Xcompiler /MT'
if ($env:EXTRA_NVCC) { $flags = "$flags $env:EXTRA_NVCC" }

Remove-Item tests\bench_pom.exe -ErrorAction SilentlyContinue   # a failed build can't run stale
cmd /c "`"$vc`" >nul 2>&1 && `"$nvcc`" $flags -o tests\bench_pom.exe tests\bench_pom.cu 2>&1"
if (-not (Test-Path tests\bench_pom.exe)) { Write-Output 'BUILD FAILED'; exit 1 }
Write-Output 'BUILD OK'

scp -i $key -P $P -o StrictHostKeyChecking=accept-new -o BatchMode=yes tests\bench_pom.exe "${h}:C:/obm/bench_pom.exe" 2>&1 | Out-Null
$cmd = if ($args.Count -ge 1) { $args -join ' ' } else { 'bench' }
ssh -i $key -p $P -o StrictHostKeyChecking=accept-new -o BatchMode=yes $h "C:\obm\bench_pom.exe $cmd"
