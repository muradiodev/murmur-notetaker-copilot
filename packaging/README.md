# Packaging Murmur for Windows

Builds a normal desktop app (its own window, no browser needed) and a `setup.exe` installer.

## What you get

- `dist\Murmur\Murmur.exe` — the desktop app. It starts the local server (127.0.0.1:8765) and
  shows the UI in its own window (WebView2). Everything runs on the machine; the only outbound calls
  are the AI summary/answer requests to the provider the user configured (Claude / OpenAI / NVIDIA).
- `dist\installer\Murmur-Setup.exe` — the installer. Installs per-user (no admin) into
  `%LOCALAPPDATA%\Programs\Murmur`, adds Start-menu and optional desktop shortcuts, and ships the
  models and ffmpeg next to the app.

User data (settings, API keys, meetings, the copilot library) is stored in
`%LOCALAPPDATA%\Murmur\data` and is **kept** when the app is uninstalled.

## One-time tools

```
.venv\Scripts\python.exe -m pip install pyinstaller pywebview
winget install JRSoftware.InnoSetup      # the setup.exe compiler
```

The target machine needs the **WebView2 runtime** (preinstalled on Windows 10/11). If missing, the
window stays blank; install "Microsoft Edge WebView2 Runtime" (Evergreen).

## Build

```
powershell -ExecutionPolicy Bypass -File packaging\build.ps1
```

- `-SkipApp` — only recompile the installer (reuse the existing `dist\Murmur`).
- `-SkipInstaller` — only build the app bundle.

Two steps run under the hood:
1. **PyInstaller** (`packaging\murmur.spec`) bundles `desktop.py`, the Python runtime, all
   dependencies and the static UI into `dist\Murmur\` (~250 MB). Models and ffmpeg are **not**
   bundled here — they are shipped by the installer.
2. **Inno Setup** (`packaging\installer.iss`) packs that bundle together with `models\` and
   `bin\ffmpeg.exe` into `dist\installer\Murmur-Setup.exe`.

The installer's Whisper models (~1.7 GB) make the setup file large. The **Parakeet** engine
(+2.9 GB, German/English, faster on CPU) is an optional component, off by default.

## How paths work in the packaged app (`app/config.py`)

- Read-only resources (`models\`, `bin\ffmpeg.exe`) sit next to `Murmur.exe`.
- Writable data goes to `%LOCALAPPDATA%\Murmur\data`.
- Two env vars override these (handy for testing or relocating the big models folder):
  `MURMUR_MODELS` and `MURMUR_DATA`.

## ⚠️ Never junction models into `dist\Murmur`

PyInstaller **deletes** `dist\Murmur` on each build. If `dist\Murmur\models` is a junction to
the real models folder, the delete recurses through it and wipes the real models. To test the exe
against the real models without copying gigabytes, use the env var instead:

```
$env:MURMUR_MODELS = "C:\path\to\Murmur-Notetaker-Copilot\models"
$env:MURMUR_DATA   = "$env:TEMP\murmur_test_data"
dist\Murmur\Murmur.exe
```

`build.ps1` also removes any leftover junction before building, as a safeguard.

## Startup log

If the packaged app fails to start, the reason is written to
`%LOCALAPPDATA%\Murmur\data\murmur.log`.
