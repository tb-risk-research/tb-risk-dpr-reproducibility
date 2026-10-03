@echo off
cd /d "%~dp0"

echo ========================================
echo   TB Risk - Diagnostic Tool
echo ========================================
echo.

if exist ".venv\Scripts\python.exe" (
    echo Using Python: .venv\Scripts\python.exe
    echo.
    set PYTHONPATH=%~dp0..
    ".venv\Scripts\python.exe" "diagnose.py"
) else (
    echo ERROR: Virtual environment not found!
    echo Please run the launcher first.
)

echo.
echo ========================================
echo   Log saved to: diagnostic_log.txt
echo ========================================
echo.
pause