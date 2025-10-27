# Medical LLM System - Complete Setup Guide

This guide will walk you through setting up the Medical Symptom Analyzer from scratch. Follow each step carefully to ensure everything works properly.

---

## Prerequisites

Before you begin, make sure you have the following installed on your computer:

- **Python 3.12.1** or higher ([Download here](https://www.python.org/downloads/))
- **PostgreSQL 14.19** or higher ([Download here](https://www.postgresql.org/download/))
- **Ollama** for running local LLM models ([Download here](https://ollama.ai/download))
- **Node.js 18+** and npm for the web interface ([Download here](https://nodejs.org/))
- **Git** for version control (optional but recommended)

---

## Step 1: PostgreSQL Database Setup

### 1.1 Install PostgreSQL

Download and install PostgreSQL from the official website. During installation:
- Set the password for the `postgres` user (remember this - you'll need it later)
- Use the default port `5432`
- Install pgAdmin 4 (recommended for database management)

### 1.2 Create the Database

Open PostgreSQL command line or pgAdmin and create a new database:

```sql
CREATE DATABASE symptom_kb;
```

You can verify the database was created by listing all databases:

```sql
\l
```

---

## Step 2: Python Environment Setup

### 2.1 Create Virtual Environment

Open PowerShell or Command Prompt, navigate to the project directory, and create a virtual environment:

```powershell
cd E:\medicalLLM\medical-llm\Healthcare-llm-system-main
python -m venv venv
```

### 2.2 Activate Virtual Environment

**On Windows (PowerShell):**
```powershell
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
venv\Scripts\activate.bat
```

**On Linux/Mac:**
```bash
source venv/bin/activate
```

You should see `(venv)` appear in your terminal prompt.

### 2.3 Install Python Dependencies

With the virtual environment activated, install all required packages:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

This will install:
- FastAPI and Uvicorn (web server)
- PostgreSQL drivers (psycopg2-binary, SQLAlchemy)
- Machine learning libraries (transformers, torch, sentence-transformers)
- Ollama Python client
- Other utilities

---

## Step 3: Ollama LLM Setup

### 3.1 Install Ollama

Download and install Ollama from [https://ollama.ai/download](https://ollama.ai/download)

### 3.2 Download the Phi Model

Open a new terminal and run:

```bash
ollama pull phi:latest
```

This downloads the Phi model (approximately 1.6GB). The download might take a few minutes depending on your internet connection.

### 3.3 Verify Ollama is Running

Check if Ollama is running:

```bash
ollama list
```

You should see the `phi:latest` model in the list.

---

## Step 4: Initialize the Knowledge Base

### 4.1 Update Database Connection

Open `knowledge_base_postgres.py` and verify the database URL matches your PostgreSQL setup:

```python
db_url = "postgresql://postgres:YOUR_PASSWORD@localhost:5432/symptom_kb"
```

Replace `YOUR_PASSWORD` with the password you set during PostgreSQL installation.

### 4.2 Run Database Initialization

This step loads all medical documents into the PostgreSQL database:

```powershell
python init_knowledge_base.py
```

When prompted, enter your PostgreSQL password. You should see output indicating:
- Connection to database successful
- 13 medical documents loaded
- Knowledge base initialization complete

---

## Step 5: Download Medical AI Model

The system uses a pre-trained medical RoBERTa model from Hugging Face.

### 5.1 First Run Download

The first time you start the API server, it will automatically download the PubMedBERT model (approximately 440MB). This is a one-time download and will be cached locally.

To pre-download the model, you can run:

```powershell
python -c "from medical_roberta_loader import load_medical_roberta_model; load_medical_roberta_model()"
```

---

## Step 6: Start the Backend API Server

### 6.1 Launch the FastAPI Server

With your virtual environment activated:

```powershell
python simple_api.py
```

You should see output like:
```
[STARTUP] Starting Symptom Analysis API...
[LOAD] Attempting to load Medical RoBERTa model...
[SUCCESS] Medical RoBERTa loaded successfully!
[SUCCESS] Pipeline initialized successfully
INFO:     Uvicorn running on http://127.0.0.1:9001
```

The API is now running on `http://127.0.0.1:9001`

### 6.2 Test the API

Open your browser and visit:
- API Documentation: `http://127.0.0.1:9001/docs`
- Health Check: `http://127.0.0.1:9001/health`

Or use curl:
```bash
curl http://127.0.0.1:9001/health
```

---

## Step 7: Setup the Web Interface

### 7.1 Navigate to Frontend Directory

Open a **new terminal window** (keep the backend running) and navigate to the web UI folder:

```powershell
cd E:\medicalLLM\medical-llm\Healthcare-llm-system-main\webui
```

### 7.2 Install Node Dependencies

```bash
npm install
```

This installs React, Vite, and other frontend dependencies.

### 7.3 Start the Development Server

```bash
npm run dev
```

You should see:
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

---

## Step 8: Access the Application

### 8.1 Open the Web Interface

Open your browser and go to:
```
http://localhost:5173
```

You should see the **Medical Symptom Analyzer** interface with:
- Green gradient background with subtle medical patterns
- Input textarea for symptom description
- "Analyze" and "Clear" buttons
- Example symptom buttons
- Medical disclaimer

### 8.2 Test the System

Try one of the example symptoms or type your own, such as:
```
I feel tightness in my chest and trouble breathing
```

Click "Analyze" and wait for the response (takes about 60-90 seconds for the first query).

---

## Step 9: Production Build (Optional)

### 9.1 Build the Frontend

When you're ready for production, build the optimized frontend:

```bash
cd webui
npm run build
```

This creates an optimized build in `webui/dist/`

### 9.2 Serve Static Files

The FastAPI backend is already configured to serve the built files. Just visit:
```
http://127.0.0.1:9001/ui
```

---

## Troubleshooting

### Database Connection Issues

**Error:** "Failed to connect to database"

**Solution:**
1. Verify PostgreSQL is running (check Windows Services or use pgAdmin)
2. Check your password in the database URL
3. Ensure the `symptom_kb` database exists
4. Check that port 5432 is not blocked by firewall

### Ollama Not Found

**Error:** "Ollama not available"

**Solution:**
1. Make sure Ollama is installed and running
2. Run `ollama list` to verify the phi model is downloaded
3. Try `ollama serve` to manually start the Ollama service

### Model Download Issues

**Error:** "Failed to load Medical RoBERTa"

**Solution:**
1. Check your internet connection
2. The first download might take several minutes (440MB)
3. If interrupted, delete the `model_cache/` folder and try again
4. System will fall back to keyword-based classification if model fails

### Port Already in Use

**Error:** "Address already in use"

**Solution:**
1. Check if another application is using port 9001 or 5173
2. Stop any existing instances of the API
3. Change the port in `simple_api.py` or `vite.config.js` if needed

### Slow Response Times

**Issue:** Analysis takes too long

**Solution:**
1. First query is always slower (model loading)
2. Subsequent queries should be faster (60-90 seconds)
3. The Phi model is CPU-optimized, GPU not required
4. Close other resource-intensive applications

---

## System Architecture

```
┌─────────────────┐
│   Web Browser   │
│ (React + Vite)  │
└────────┬────────┘
         │ HTTP
         ▼
┌─────────────────┐
│   FastAPI       │
│   Backend       │
│   Port 9001     │
└────────┬────────┘
         │
    ┌────┴────┬────────────┬──────────┐
    │         │            │          │
    ▼         ▼            ▼          ▼
┌────────┐ ┌──────┐ ┌──────────┐ ┌────────┐
│PubMed  │ │Ollama│ │PostgreSQL│ │Medical │
│BERT    │ │ Phi  │ │symptom_kb│ │  KB    │
│Classify│ │ LLM  │ │  13 docs │ │ JSON   │
└────────┘ └──────┘ └──────────┘ └────────┘
```

---

## Quick Command Reference

### Start Backend
```powershell
cd E:\medicalLLM\medical-llm\Healthcare-llm-system-main
.\venv\Scripts\Activate.ps1
python simple_api.py
```

### Start Frontend (Development)
```powershell
cd E:\medicalLLM\medical-llm\Healthcare-llm-system-main\webui
npm run dev
```

### View Logs
Backend logs appear in the terminal where you ran `simple_api.py`

### Stop Servers
Press `Ctrl+C` in the respective terminal windows

---

## Next Steps

Once everything is running:

1. **Test thoroughly** - Try various symptom descriptions
2. **Monitor performance** - Check response times and accuracy
3. **Review logs** - Look for any warnings or errors
4. **Customize** - Adjust the UI colors, add more medical documents, or fine-tune prompts
5. **Deploy** - Consider containerizing with Docker for production deployment

---

## Support

For issues or questions:
- Check the troubleshooting section above
- Review the error logs in the terminal
- Consult the `README.md` for additional documentation
- Check Ollama documentation: [https://ollama.ai/docs](https://ollama.ai/docs)
- Check FastAPI documentation: [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

---

**Last Updated:** October 2025  
**Project Version:** 1.0  
**Python Version:** 3.12.1  
**PostgreSQL Version:** 14.19
