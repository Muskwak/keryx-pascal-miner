$log = "C:\obm\gpulog.csv"
"timestamp,util_gpu,util_mem,mem_used_mb,temp_c,power_w" | Out-File $log
for ($i=0; $i -lt 150; $i++) {
  $ts = (Get-Date -Format "HH:mm:ss")
  $nvs = & "C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe" --query-gpu=utilization.gpu,utilization.memory,memory.used,temperature.gpu,power.draw --format=csv,noheader,nounits 2>$null
  if ($nvs) {
    "$ts,$nvs" | Out-File $log -Append
  } else {
    "$ts,---,---,---,---,---" | Out-File $log -Append
  }
  Start-Sleep -Seconds 2
}
