@echo off
echo ======================================
echo Starting ViXNet Web Application
echo ======================================
echo.

REM Check if we're in the correct directory
if not exist "backend" (
    echo Error: Please run this script from the web_app directory
    echo    cd ViXNet\web_app
    echo    start.bat
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed
    exit /b 1
)

REM Check Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed
    exit /b 1
)

echo Python found
echo Node.js found
echo.

REM Install backend dependencies
echo Checking backend dependencies...
cd backend
if not exist ".installed" (
    echo    Installing Python packages...
    pip install -r requirements.txt
    type nul > .installed
    echo    Backend dependencies installed
) else (
    echo    Backend dependencies already installed
)
cd ..
echo.

REM Install frontend dependencies
echo Checking frontend dependencies...
cd frontend
if not exist "node_modules" (
    echo    Installing Node.js packages...
    call npm install
    echo    Frontend dependencies installed
) else (
    echo    Frontend dependencies already installed
)
cd ..
echo.

echo ======================================
echo Starting Services
echo ======================================
echo.

REM Start backend
echo Starting Flask backend on http://localhost:5000...
cd backend
start /B python app.py > backend.log 2>&1
cd ..
echo.

REM Wait for backend
timeout /t 3 /nobreak >nul

REM Start frontend
echo Starting React frontend on http://localhost:3000...
echo.
cd frontend
npm start
