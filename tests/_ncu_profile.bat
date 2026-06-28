@echo off
cd /d "C:\Users\KFN Collegiate\Downloads\keryx-miner-v0.3.3-OPoI-win64-amd64"
"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe" --set basic -k regex:pom_mine -c 50 -f -o C:\obm\pom_profile keryx-miner.exe --mining-address keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3 -s 127.0.0.1 -p 22110 --light
