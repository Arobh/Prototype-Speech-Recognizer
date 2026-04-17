@echo off
setlocal
cd /d "%~dp0"

set MODEL_DIR=models\vosk-model-small-en-us-0.15
set ZIP_PATH=models\vosk-model-small-en-us-0.15.zip
set MODEL_URL=https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

if exist "%MODEL_DIR%" (
  echo Vosk model already exists at %MODEL_DIR%.
  exit /b 0
)

if not exist "models" mkdir models

echo Downloading Vosk small English model...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Invoke-WebRequest -Uri '%MODEL_URL%' -OutFile '%ZIP_PATH%'"

echo Extracting Vosk model...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Expand-Archive -LiteralPath '%ZIP_PATH%' -DestinationPath 'models' -Force"

del "%ZIP_PATH%"
echo Vosk model ready at %MODEL_DIR%.
