@echo off
setlocal
cd /d "%~dp0"

if exist "%~dp0venv\python.exe" (
  set "PY=%~dp0venv\python.exe"
) else (
  set "PY=python"
)

if not defined DEEPSEEK_API_KEY (
  echo.
  echo [Ideogram Studio] DEEPSEEK_API_KEY nije postavljen.
  echo   set DEEPSEEK_API_KEY=sk-...
  echo   ili upisi kljuc u ideogram_studio\config.json
  echo.
)

echo Starting Ideogram Studio...
"%PY%" -m ideogram_studio %*
set "ERR=%ERRORLEVEL%"
if not "%ERR%"=="0" (
  echo.
  echo Ideogram Studio exited with code %ERR%.
  pause
)
endlocal & exit /b %ERR%
