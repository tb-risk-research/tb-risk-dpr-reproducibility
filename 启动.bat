@echo off
cd /d "%~dp0"

echo ========================================
echo   TB Risk Assessment System
echo ========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [First Run] Creating virtual environment...
    echo.
    python -m venv .venv
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to create venv.
        echo.
        pause
        exit /b 1
    )
    echo Installing dependencies...
    echo.
    ".venv\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
    ".venv\Scripts\python.exe" -m pip install -e ".[full]" -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        ".venv\Scripts\python.exe" -m pip install -e ".[full]"
    )
    if errorlevel 1 (
        echo.
        echo ERROR: Failed to install dependencies.
        echo.
        pause
        exit /b 1
    )
    echo.
    echo Done!
    echo.
)

echo Starting application...
echo.
set PYTHONPATH=%~dp0..
".venv\Scripts\python.exe" "_launch_gui.py"

echo.
echo Application exited with code: %errorlevel%
echo.
pause