@echo off
setlocal

cd /d "%~dp0"

REM Prefer Python 3.11 venv
if not exist .venv311 (
  py -3.11 -m venv .venv311
)

call .venv311\Scripts\python -m pip install --upgrade pip
call .venv311\Scripts\python -m pip install -r requirements.txt

REM Start server
.venv311\Scripts\python -m uvicorn main:app --host 127.0.0.1 --port 8000 --app-dir backend

endlocal
