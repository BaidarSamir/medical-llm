# Web UI (React + Vite)

Quick frontend for the Medical Symptom Analyzer.

Run locally (dev):

1. cd webui
2. npm install
3. npm run dev

This starts Vite on http://localhost:5173. During development the frontend will POST to `/analyze` on the same host; if running the backend separately, either run the backend on port 5173 proxy or set up CORS and change the axios base URL in `src/App.jsx`.

Build for production:

1. npm run build
2. Copy `dist/` into the backend `webui/dist` or run `uvicorn simple_api:app --reload` and serve files from `webui/dist` (FastAPI should be configured to mount static files).

Notes:
- Lightweight design, plain CSS for simplicity
- Buttons for copy/download JSON
