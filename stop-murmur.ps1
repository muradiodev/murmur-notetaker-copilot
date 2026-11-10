# Stops the Murmur server (the process listening on port 8765).
$conns = Get-NetTCPConnection -LocalPort 8765 -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conns) { Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue }
