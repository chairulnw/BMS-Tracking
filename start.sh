#!/bin/bash
# Jalankan semua service BMS-IIP sekaligus.
# Setiap service dibuka di terminal window terpisah.

ROOT="$(cd "$(dirname "$0")" && pwd)"

open_tab() {
  osascript -e "tell application \"Terminal\" to do script \"$1\""
}

echo "Starting BMS-IIP services..."

open_tab "cd '$ROOT/backend'         && source .venv/bin/activate && uvicorn app.main:app --port 8002 --reload"
open_tab "cd '$ROOT/ai-service'      && source .venv/bin/activate && uvicorn app.main:app --port 8001 --reload"
open_tab "cd '$ROOT/sistem-kamera-pengawas' && ng serve"

echo "Done — 3 terminal windows opened."
echo "  Backend:  http://localhost:8002/docs"
echo "  AI:       http://localhost:8001/docs"
echo "  Frontend: http://localhost:4200"
