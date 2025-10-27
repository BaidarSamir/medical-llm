# Medical Symptom Analysis – Local, Private, and Practical

This is a local-first symptom analysis system I built end-to-end. You type your symptoms in plain English, and it suggests the most relevant medical department and explains the reasoning in clear, friendly language. Everything runs on your machine—no cloud APIs for your inputs.

## What’s inside (what I actually used)

- Backend: FastAPI (Python) serving a single endpoint for symptom analysis
- Knowledge base: PostgreSQL storing curated medical JSON documents from `KB_/`
	- Optional pgvector; if not available, it falls back to keyword search (LIKE)
- Retrieval: Simple RAG-style retrieval using the knowledge base results
- LLM: Local model via Ollama (default model name: `phi:latest`)
- Classifier: PubMedBERT loader present, but current pipeline relies mainly on retrieval + LLM
- Frontend: React + Vite app in `webui/` with a soft green medical theme

All core code lives in `Healthcare-llm-system-main/`.

## How to run (quick version)

For detailed steps with screenshots and troubleshooting, see `SETUP_GUIDE.md`.

1) Python deps

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r Healthcare-llm-system-main/requirements.txt
```

2) PostgreSQL knowledge base

- Create a database (default in code): `symptom_kb`
- pgvector is optional. If it isn’t enabled, the app falls back to text search.

Initialize tables and load the JSON knowledge base:

```powershell
cd Healthcare-llm-system-main
python init_knowledge_base.py
```

3) Ollama model (runs locally)

Install Ollama from https://ollama.ai and pull a lightweight model:

```powershell
ollama pull phi
```

4) Start the backend API

```powershell
python Healthcare-llm-system-main/simple_api.py
```

The API runs at: http://127.0.0.1:9001

5) Start the frontend (optional but recommended)

```powershell
cd Healthcare-llm-system-main/webui
npm install
npm run dev
```

Open http://localhost:5173 in your browser.

## API endpoints (backend on port 9001)

- POST `/analyze` – Submit symptom text and receive department + explanation
- GET `/health` – Basic health check
- GET `/departments` – List of available departments
- GET `/docs` – Interactive Swagger UI

## How it works (brief)

1) You enter symptoms in the UI (or POST to `/analyze`).
2) The pipeline retrieves the most relevant document from PostgreSQL.
	 - If pgvector is available, vector similarity is used; otherwise, keyword search.
3) A prompt is built from your input + the retrieved medical document.
4) A local LLM (via Ollama, default `phi:latest`) generates a clear, empathetic explanation.
5) The API returns a JSON object with the suggested department and explanation.

## Project structure (actual)

```
Healthcare-llm-system-main/
	KB_/                         # Curated medical JSON documents (the knowledge base)
	complete_pipeline.py         # Orchestrates retrieval → prompt → LLM
	init_knowledge_base.py       # Creates tables and loads KB into PostgreSQL
	knowledge_base_postgres.py   # DB layer with optional pgvector, LIKE fallback
	local_llm.py                 # Ollama/TextGenWebUI/llama.cpp backends (uses Ollama by default)
	medical_roberta_loader.py    # PubMedBERT loader (present but not required to run)
	prompt_builder.py            # Builds the final prompt for the LLM
	rag_retriever.py             # Retrieves relevant docs from the KB
	simple_api.py                # FastAPI app (port 9001)
	webui/                       # React + Vite frontend (dev server on 5173)
```

There are also training and experimental scripts kept for history. They’re not needed to run the app, but they show how I got here.

## Notes and limitations

- Privacy: inputs stay local; the LLM and DB run on your machine.
- Performance: first run may be slower while models load; typical responses take tens of seconds on CPU.
- Scope: the knowledge base includes 13 symptom documents; it’s a helpful guide—not a diagnosis tool.
- Database: works with or without pgvector; text search is the default fallback.

## Safety disclaimer

This project is for educational purposes and should not be used as a substitute for professional medical advice, diagnosis, or treatment. For emergencies or serious symptoms, seek immediate medical care.

## Links

- Setup guide: `Healthcare-llm-system-main/SETUP_GUIDE.md`
- Requirements: `Healthcare-llm-system-main/requirements.txt`
- Frontend: `Healthcare-llm-system-main/webui/`

— Built to be practical, private, and understandable.

## Screenshot

Below is a sample of the UI while analyzing symptoms.

![Medical Symptom Analyzer UI](docs/Example.png)

