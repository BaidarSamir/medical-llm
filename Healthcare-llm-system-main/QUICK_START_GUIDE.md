# 🚀 Quick Start Guide - Medical LLM System

## ✅ Current Status
- ✅ Virtual environment created and activated `(venv)`
- ⏳ Installing packages (tensorflow, sentence-transformers)...

---

## Step-by-Step Instructions

### **Step 1: Activate Virtual Environment** ✅ COMPLETED
```powershell
cd E:\medicalLLM\medical-llm\Healthcare-llm-system-main

# Set execution policy (one-time per terminal session)
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force

# Activate virtual environment
.\venv\Scripts\Activate.ps1
```

You should see `(venv)` at the beginning of your prompt.

---

### **Step 2: Install Dependencies** ⏳ IN PROGRESS
The packages are being installed in the background. Wait for the installation to complete.

To manually install if needed:
```powershell
pip install fastapi uvicorn psycopg2-binary sqlalchemy pydantic requests numpy pandas
pip install --default-timeout=100 tensorflow sentence-transformers python-multipart
```

---

### **Step 3: Set Up PostgreSQL Database**

#### Option A: If PostgreSQL is Already Installed
```powershell
# Start PostgreSQL service (if not running)
# Then create database and enable pgvector

# Open psql
psql -U postgres

# In psql, run:
CREATE DATABASE symptom_kb;
\c symptom_kb
CREATE EXTENSION vector;
\q
```

#### Option B: If PostgreSQL is NOT Installed
1. Download PostgreSQL 14+ from: https://www.postgresql.org/download/windows/
2. Install it (remember the postgres user password!)
3. Follow Option A above

---

### **Step 4: Initialize the Knowledge Base**

Load the medical documents into the database:

```powershell
# Make sure you're in the project directory with (venv) active
python knowledge_base_postgres.py
```

Expected output:
```
✅ PostgreSQL knowledge base initialized
📚 Loading 13 medical documents...
✅ Knowledge base initialized successfully
```

---

### **Step 5: Install and Configure Ollama (for Local LLM)**

#### A. Download and Install Ollama
- Visit: https://ollama.ai/download
- Download the Windows installer
- Run the installer

#### B. Start Ollama Service
```powershell
# In a NEW terminal (keep the current one for API server)
ollama serve
```

Keep this terminal running!

#### C. Download the Mistral 7B Model
```powershell
# In ANOTHER new terminal
ollama pull mistral:7b-instruct
```

This will download ~4GB. Wait for completion.

#### D. Test Ollama (Optional)
```powershell
ollama run mistral:7b-instruct "Hello, how are you?"
```

---

### **Step 6: Start the API Server**

Back in your main terminal (with `(venv)` active):

```powershell
python simple_api.py
```

Expected output:
```
✅ Complete pipeline initialized successfully
INFO:     Started server process
INFO:     Uvicorn running on http://127.0.0.1:9000
```

---

### **Step 7: Test the System**

#### Option A: Use Web Browser
1. Open browser: `http://127.0.0.1:9000/docs`
2. Click on **POST /analyze**
3. Click **"Try it out"**
4. Enter test data:
   ```json
   {
     "symptom_text": "I feel tightness in my chest and I'm having trouble breathing"
   }
   ```
5. Click **"Execute"**

#### Option B: Use PowerShell
```powershell
# Open a NEW PowerShell terminal
$body = @{
    symptom_text = "I feel tightness in my chest and I'm having trouble breathing"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:9000/analyze" -Body $body -ContentType "application/json"
```

#### Option C: Use Python Script
Create `test_api.py`:
```python
import requests

response = requests.post(
    "http://127.0.0.1:9000/analyze",
    json={"symptom_text": "I feel tightness in my chest and I'm having trouble breathing"}
)

print(response.json())
```

Run it:
```powershell
python test_api.py
```

---

## Expected Output

You should get a JSON response like:
```json
{
  "department": "Cardiology",
  "description": "Hello, I understand you're experiencing chest tightness and difficulty breathing. These symptoms can be concerning and should be taken seriously. Based on the information available, this could be related to various heart conditions such as angina, arrhythmia, or other cardiovascular issues..."
}
```

---

## 🐛 Troubleshooting

### Issue: Activation Script Won't Run
```powershell
# Solution: Set execution policy first
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
```

### Issue: PostgreSQL Connection Failed
- Check if PostgreSQL is running: `pg_isready`
- Verify database exists: `psql -U postgres -l`
- Check connection string in `knowledge_base_postgres.py`

### Issue: Ollama Connection Failed
- Check Ollama is running: Visit `http://localhost:11434`
- The system will fall back to mock responses if Ollama is unavailable

### Issue: Port 9000 Already in Use
- Change the port in `simple_api.py` (line with `uvicorn.run()`)
- Or kill the process using port 9000:
  ```powershell
  netstat -ano | findstr :9000
  taskkill /PID <PID> /F
  ```

### Issue: Package Installation Timeout
```powershell
# Use longer timeout
pip install --default-timeout=100 <package_name>
```

---

## 📝 Additional API Endpoints

- `GET /` - API information
- `GET /health` - Health check
- `GET /departments` - List available departments
- `GET /docs` - Interactive API documentation (Swagger UI)

---

## 🎯 Architecture Flow

```
User Input
    ↓
RoBERTa Classifier
    ↓
RAG Retriever (PostgreSQL + pgvector)
    ↓
Prompt Builder
    ↓
Local LLM (Mistral 7B via Ollama)
    ↓
JSON Response: {department, description}
```

---

## 💡 Tips

1. **Keep Ollama running** in a separate terminal when using the API
2. **Use the `/docs` endpoint** for easy testing via browser
3. **Check logs** if something goes wrong - they're printed in the terminal
4. **PostgreSQL must be running** for the knowledge base to work
5. **First LLM response** may take longer (model loading)

---

## 🔄 Restarting the System

After a reboot or when reopening VS Code:

```powershell
# 1. Navigate to project
cd E:\medicalLLM\medical-llm\Healthcare-llm-system-main

# 2. Activate venv
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process -Force
.\venv\Scripts\Activate.ps1

# 3. Start Ollama (in separate terminal)
ollama serve

# 4. Start API server
python simple_api.py
```

---

**Need Help?** Check the error messages in the terminal for specific issues!
