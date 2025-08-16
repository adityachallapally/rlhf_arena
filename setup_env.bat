@echo off
REM RLHF Arena Environment Setup Script for Windows
REM This script sets up the environment for RLHF Arena

echo 🚀 Setting up RLHF Arena Environment
echo =====================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not available
    echo Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo ✅ Python detected

REM Check if pip is available
pip --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: pip is not available
    echo Please install pip and try again
    pause
    exit /b 1
)

echo ✅ pip detected

REM Create virtual environment
echo 🔧 Creating virtual environment...
python -m venv rlhf_arena_env

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call rlhf_arena_env\Scripts\activate.bat

REM Upgrade pip
echo 🔧 Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install dependencies
echo 📦 Installing dependencies...
pip install -r requirements.txt

REM Install package in development mode
echo 🔧 Installing RLHF Arena in development mode...
pip install -e .

REM Create necessary directories
echo 📁 Creating necessary directories...
if not exist experiments mkdir experiments
if not exist reports mkdir reports
if not exist checkpoints mkdir checkpoints
if not exist logs mkdir logs
if not exist cache mkdir cache
if not exist archives mkdir archives
if not exist exports mkdir exports

echo.
echo 🎉 Environment setup completed successfully!
echo.
echo To activate the environment, run:
echo   rlhf_arena_env\Scripts\activate.bat
echo.
echo To run tests:
echo   python run_tests.py
echo.
echo To run a quick start example:
echo   python examples\quick_start.py
echo.
echo To check system information:
echo   python cli.py info
echo.
echo For more information, see README.md
pause 