@echo off
echo ========================================
echo Personality Predictor Setup Script
echo ========================================
echo.

echo Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo Python found! Installing dependencies...
pip install -r requirements.txt

echo.
echo Creating database...
python manage.py makemigrations
python manage.py migrate

echo.
echo Setup complete! Run the application with:
echo python manage.py runserver
echo.
echo Then open http://127.0.0.1:8000 in your browser
pause 