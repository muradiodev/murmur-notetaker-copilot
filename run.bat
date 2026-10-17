@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Evvelce setup.bat-i isled.
  pause
  exit /b 1
)
echo Murmur Notetaker http://127.0.0.1:8765 unvaninda acilir. Bu pencereni baglama.
.venv\Scripts\python.exe -m app.server
pause
