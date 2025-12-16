#!/bin/bash
# Start script for LoRA Attention Visualizer
# Kills existing processes on ports 9001/9002 and starts fresh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

echo "=== LoRA Attention Visualizer ==="
echo ""

# Kill any existing processes on our ports
echo "Killing processes on ports 9001 and 9002..."
lsof -ti:9001 | xargs kill -9 2>/dev/null || true
lsof -ti:9002 | xargs kill -9 2>/dev/null || true
sleep 1

# Deploy Modal function
echo "Deploying Modal function..."
cd "$SCRIPT_DIR/modal"
if [ ! -d ".venv" ]; then
    echo "Creating modal virtual environment..."
    uv venv
    source .venv/bin/activate
    uv pip install modal
else
    source .venv/bin/activate
fi
modal deploy attention.py 2>&1 | tee "$LOG_DIR/modal-deploy.log"
deactivate
cd "$SCRIPT_DIR"
echo "Modal function deployed."
echo ""

# Start backend
echo "Starting backend on port 9002..."
cd "$SCRIPT_DIR/backend"
if [ ! -d ".venv" ]; then
    echo "Creating backend virtual environment..."
    uv venv
fi
source .venv/bin/activate
uv pip install -r requirements.txt --quiet 2>/dev/null || true
uvicorn app:app --host 0.0.0.0 --port 9002 2>&1 | tee "$LOG_DIR/backend.log" &
BACKEND_PID=$!
cd "$SCRIPT_DIR"

# Wait for backend to be ready
echo "Waiting for backend to start..."
sleep 2

# Start frontend
echo "Starting frontend on port 9001..."
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
    echo "Installing frontend dependencies..."
    npm install
fi
npm run dev 2>&1 | tee "$LOG_DIR/frontend.log" &
FRONTEND_PID=$!
cd "$SCRIPT_DIR"

echo ""
echo "=== Services Started ==="
echo "Frontend: http://localhost:9001"
echo "Backend:  http://localhost:9002"
echo "Modal:    Deployed (logs via 'modal app logs lora-attention-visualizer')"
echo ""
echo "Logs: $LOG_DIR/"
echo "  - backend.log"
echo "  - frontend.log"
echo "  - modal-deploy.log"
echo ""
echo "Press Ctrl+C to stop all services"

# Trap Ctrl+C to clean up
trap "echo 'Stopping services...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for either process to exit
wait
