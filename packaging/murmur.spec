# PyInstaller spec for the Murmur desktop app (onedir).
# Build from the project root:  pyinstaller packaging/murmur.spec --noconfirm
#
# The heavy, read-only payloads (models/, bin/ffmpeg.exe) are NOT bundled here; the Inno Setup
# installer ships them next to Murmur.exe. config.py looks for them beside the executable.

from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_submodules

ROOT = Path(SPECPATH).resolve().parent          # project root (packaging/ is one below)

datas = [(str(ROOT / "app" / "static"), "app/static")]
binaries = []
hiddenimports = ["app.server", "app.transcriber", "app.diarize", "app.audio", "app.dtln",
                 "app.aec", "app.names", "app.summary", "app.copilot", "app.storage",
                 "app.setup_models"]

# Native / data-carrying packages: pull their binaries and data files in full.
for pkg in ["ctranslate2", "faster_whisper", "sherpa_onnx", "onnxruntime", "soxr",
            "soundfile", "soundcard", "anthropic", "openai", "av"]:
    try:
        d, b, h = collect_all(pkg)
        datas += d
        binaries += b
        hiddenimports += h
    except Exception as exc:                     # a package that is not installed is simply skipped
        print(f"[spec] collect_all({pkg}) skipped: {exc}")

# uvicorn resolves its loop/protocol/lifespan implementations by string at runtime.
hiddenimports += collect_submodules("uvicorn")

excludes = ["tkinter", "matplotlib", "torch", "torchaudio", "torchvision", "pandas",
            "scipy", "notebook", "IPython", "pytest", "PIL", "cv2"]

a = Analysis(
    [str(ROOT / "desktop.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Murmur",
    debug=False,
    strip=False,
    upx=False,
    console=False,                               # windowed app: no console popup
    icon=str(ROOT / "brand" / "kit" / "favicon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Murmur",
)
