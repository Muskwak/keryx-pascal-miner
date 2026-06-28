Set WshShell = CreateObject("WScript.Shell")
WshShell.CurrentDirectory = "C:\Users\KFN Collegiate\Downloads\keryx-miner-v0.3.3-OPoI-win64-amd64"
WshShell.Run """keryx-miner.exe"" --mining-address keryx:qpcptntu45n0xtyq60apnwnhpkta0ujzt5sy3uk5v6nrjvxlqhamjyc882jj3 -s 127.0.0.1 -p 22110 --light", 0, False
