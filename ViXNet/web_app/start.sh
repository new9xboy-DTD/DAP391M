#!/bin/bash

echo "======================================"
echo "🚀 Starting ViXNet Web Application"
echo "======================================"
echo ""

# Check if we're in the correct directory
if [ ! -d "backend" ] || [ ! -d "frontend" ]; then
    echo "❌ Error: Please run this script from the web_app directory"
    echo "   cd ViXNet/web_app && ./start.sh"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "❌ Error: Python is not installed"
    exit 1
fi

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed"
    exit 1
fi

echo "✅ Python found: $(python3 --version 2>/dev/null || python --version)"
echo "✅ Node.js found: $(node --version)"
echo ""

# Install backend dependencies if needed
echo "📦 Checking backend dependencies..."
if [ ! -f "backend/.installed" ]; then
    echo "   Installing Python packages..."
    cd backend
    pip install -r requirements.txt
    touch .installed
    cd ..
    echo "   ✅ Backend dependencies installed"
else
    echo "   ✅ Backend dependencies already installed"
fi
echo ""

# Install frontend dependencies if needed
echo "📦 Checking frontend dependencies..."
if [ ! -d "frontend/node_modules" ]; then
    echo "   Installing Node.js packages..."
    cd frontend
    npm install
    cd ..
    echo "   ✅ Frontend dependencies installed"
else
    echo "   ✅ Frontend dependencies already installed"
fi
echo ""

echo "======================================"
echo "🎯 Starting Services"
echo "======================================"
echo ""

# Start backend in background
echo "🔧 Starting Flask backend on http://localhost:5000..."
cd backend
python3 app.py > backend.log 2>&1 &
BACKEND_PID=$!
cd ..
echo "   Backend PID: $BACKEND_PID"
echo ""

# Wait for backend to start
sleep 3

# Start frontend
echo "🎨 Starting React frontend on http://localhost:3000..."
echo ""
cd frontend
npm start

# Cleanup on exit
echo ""
echo "Shutting down services..."
kill $BACKEND_PID 2>/dev/null
echo "✅ Services stopped"
