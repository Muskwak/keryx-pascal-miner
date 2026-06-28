@echo off
cd /d "C:\Users\KFN Collegiate\Downloads\keryx-miner-v0.3.3-OPoI-win64-amd64"
where nsys 2>nul
if errorlevel 1 echo nsys not found & exit /b 1
nsys profile -t cuda,nvtx -o C:\obm\pom_timeline -d 300 -f true -w true keryx-miner.exe --mining-address keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3 -s 127.0.0.1 -p 22110 --light
