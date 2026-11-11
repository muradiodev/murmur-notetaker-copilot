# Stops the Murmur server (the process listening on port 8765) together with its helper processes
# (speaker voiceprints and labeling run in child processes). The venv's python.exe is only a launcher
# for the real interpreter, so it is stopped too.
$conns = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($c.OwningProcess)" -ErrorAction SilentlyContinue
    & taskkill.exe /PID $c.OwningProcess /T /F 2>&1 | Out-Null
    if ($proc -and $proc.ParentProcessId) {
        $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.ParentProcessId)" -ErrorAction SilentlyContinue
        if ($parent -and $parent.Name -eq "python.exe") { Stop-Process -Id $parent.ProcessId -Force -ErrorAction SilentlyContinue }
    }
}
