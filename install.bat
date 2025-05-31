@echo off
setlocal enabledelayedexpansion

echo ===================================
echo TTS voice clone
echo ===================================

:: Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python not found! Please install Python 3.12 or later.
    echo You can download it from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2" %%a in ('python -c "import sys; print(sys.version.split()[0])"') do set PYTHON_VERSION=%%a
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

:: Check if virtual environment exists
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if !ERRORLEVEL! neq 0 (
        echo ERROR: Failed to create virtual environment.
        echo Make sure Python is installed and available in PATH.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

:: Activate virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if !ERRORLEVEL! neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

:: Check if requirements.txt exists
if not exist "requirements.txt" (
    echo WARNING: requirements.txt not found in current directory.
    echo Please make sure requirements.txt exists.
    pause
    exit /b 1
)

:: Install or update dependencies
echo Installing/updating dependencies from requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

echo Dependencies installed successfully.
echo.

:: Check if main.py exists
if not exist "tts_clone_inference\main.py" (
    echo ERROR: tts_clone_inference\main.py not found.
    echo Please make sure the file exists in the correct location.
    pause
    exit /b 1
)

:: Run the main script with --help
echo Running TTS Clone Inference with --help...
echo ============================================
python tts_clone_inference\main.py --help
echo ============================================

echo.
echo Script execution completed.


pause
