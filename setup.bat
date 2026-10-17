@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Python muhiti yaradilir...
  python -m venv .venv || (echo Python 3.10+ lazimdir & pause & exit /b 1)
)
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt || (pause & exit /b 1)
.venv\Scripts\python.exe -m app.setup_models
pause
