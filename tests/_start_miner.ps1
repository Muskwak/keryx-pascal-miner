$exe = "C:\Users\KFN Collegiate\Downloads\keryx-miner-v0.3.3-OPoI-win64-amd64\keryx-miner.exe"
$args = "--mining-address keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3 -s 127.0.0.1 -p 22110 --light"
$log = "C:\obm\miner_bg.log"
Start-Process -NoNewWindow -FilePath $exe -ArgumentList $args -RedirectStandardOutput $log -RedirectStandardError "${log}.err"
