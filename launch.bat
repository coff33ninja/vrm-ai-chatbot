@echo off
echo Starting VRM AI Chatbot...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.11+ from https://python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Run the application
echo.
echo Launching VRM AI Chatbot...
python main.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo Application exited with error. Check logs for details.
    pause
)
