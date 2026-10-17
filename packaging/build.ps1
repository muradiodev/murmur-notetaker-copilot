# Build Murmur: the PyInstaller desktop bundle, then the Windows installer (setup.exe).
#
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1
#   powershell -ExecutionPolicy Bypass -File packaging\build.ps1 -SkipApp     # only re-compile the installer
#
# Output: dist\Murmur\ (the app) and dist\installer\Murmur-Setup.exe

param(
    [switch]$SkipApp,          # skip PyInstaller, only rebuild the installer
    [switch]$SkipInstaller     # only build the app bundle
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "Virtual env not found. Run setup.bat first." }

# Guard: never let PyInstaller delete models/ffmpeg through a leftover test junction.
foreach ($name in @("models", "bin")) {
    $j = Join-Path $root "dist\Murmur\$name"
    if (Test-Path $j) {
        $item = Get-Item $j -Force
        if ($item.LinkType) { cmd /c rmdir "$j" | Out-Null; Write-Host "Removed test junction: $name" }
    }
}

if (-not $SkipApp) {
    Write-Host "== Building the app bundle (PyInstaller) =="
    & $py -m PyInstaller "packaging\murmur.spec" --noconfirm --distpath dist --workpath build\pyi
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }
}

if (-not $SkipInstaller) {
    $iscc = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe",
        "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
    ) | Where-Object { Test-Path $_ } | Select-Object -First 1
    if (-not $iscc) { throw "Inno Setup (ISCC.exe) not found. Install it: winget install JRSoftware.InnoSetup" }

    Write-Host "== Building the installer (Inno Setup) =="
    & $iscc "packaging\installer.iss"
    if ($LASTEXITCODE -ne 0) { throw "ISCC failed" }
    Write-Host "Installer: dist\installer\Murmur-Setup.exe"
}

Write-Host "Done."
