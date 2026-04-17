@echo off
setlocal
set PORT=5500

for /f "tokens=5" %%p in ('netstat -ano ^| findstr /r /c:":%PORT% .*LISTENING"') do (
  echo Stopping process %%p using port %PORT%.
  taskkill /PID %%p /F
  exit /b 0
)

echo No server is using port %PORT%.
