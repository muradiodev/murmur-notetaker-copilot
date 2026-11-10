# Starts Murmur hidden (if not running yet), waits until it is ready, then opens it in the browser.
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$url = "http://127.0.0.1:8765"
function Test-Up { try { Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null; $true } catch { $false } }
if (-not (Test-Up)) {
    $env:MURMUR_NO_BROWSER = "1"
    Start-Process -FilePath "$dir\.venv\Scripts\python.exe" -ArgumentList "-m","app.server" -WorkingDirectory $dir -WindowStyle Hidden `
        -RedirectStandardError "$dir\data\launcher.err.log" -RedirectStandardOutput "$dir\data\launcher.out.log"
    for ($i = 0; $i -lt 90 -and -not (Test-Up); $i++) { Start-Sleep -Seconds 1 }
}
Start-Process $url
