@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set PORT=
for /l %%p in (5500,1,5510) do (
  netstat -ano | findstr /r /c:":%%p .*LISTENING" >nul
  if errorlevel 1 (
    set PORT=%%p
    goto :found_port
  )
)

echo No free port found between 5500 and 5510.
pause
exit /b 1

:found_port

echo Starting Prototype Speech Recognizer on http://127.0.0.1:%PORT%

if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port %PORT%
) else (
  python -m uvicorn server:app --host 127.0.0.1 --port %PORT%
)
