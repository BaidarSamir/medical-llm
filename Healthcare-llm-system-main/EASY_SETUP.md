# 🎯 EASY SETUP - No PostgreSQL Required!

## 📋 What You Need to Know

Your Medical LLM system can work **WITHOUT PostgreSQL**! I've created a simpler version that uses file-based knowledge base.

---

## ✅ Current Progress

- ✅ Virtual environment created
- ✅ Virtual environment activated
- ⏳ Python packages installing (wait for completion)
- ✅ Simple API created (`simple_api_no_db.py`)

---

## 🚀 Quick Start (3 Simple Steps!)

### **Step 1: Wait for Package Installation** ⏳

The packages are still installing in the background terminal. Wait until you see:
```
Successfully installed tensorflow sentence-transformers ...
```

### **Step 2: Start the Simple API Server** 

Once packages are installed, run:

```powershell
# Make sure you're in the project directory with (venv) active
cd E:\medicalLLM\medical-llm\Healthcare-llm-system-main
py simple_api_no_db.py
```

Expected output:
```
🚀 Starting Simple Medical Symptom Analysis API...
📚 Using file-based knowledge base (no PostgreSQL required)
✅ Loaded 13 medical documents
🌐 Server will be available at: http://127.0.0.1:9000
```

### **Step 3: Test the System**

Open your browser and go to:
```
http://127.0.0.1:9000/docs
```

You'll see an interactive API interface where you can:
1. Click **POST /analyze**
2. Click **"Try it out"**
3. Enter symptoms like: `"I feel tightness in my chest and I'm having trouble breathing"`
4. Click **"Execute"**
5. See the results!

---

## 🧪 Quick Test Commands

### Test in PowerShell:
```powershell
# Test 1: Health check
Invoke-RestMethod -Uri "http://127.0.0.1:9000/health"

# Test 2: Analyze symptoms
$body = @{
    symptom_text = "I feel tightness in my chest and I'm having trouble breathing"
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:9000/analyze" -Body $body -ContentType "application/json"
```

---

## 📊 What This Simple Version Does

✅ **Works without PostgreSQL** - Uses JSON files from `KB_/` folder  
✅ **13 Medical documents** - Covers common symptoms  
✅ **Keyword matching** - Fast and simple symptom detection  
✅ **Department classification** - Cardiology vs General Medicine  
✅ **REST API** - Easy to test and integrate  

❌ **No vector search** - Uses simpler keyword matching  
❌ **No LLM integration** - Returns pre-written medical advice  

---

## 🔄 If You Want Full Features Later

To get the **full version** with PostgreSQL + Ollama LLM:

### Install PostgreSQL:
1. Download from: https://www.postgresql.org/download/windows/
2. Install (use default settings)
3. Remember the password you set for `postgres` user
4. After install, run:
   ```powershell
   # Open psql
   psql -U postgres
   
   # In psql:
   CREATE DATABASE symptom_kb;
   \c symptom_kb
   CREATE EXTENSION vector;
   \q
   ```

### Initialize knowledge base:
```powershell
py knowledge_base_postgres.py
```

### Install Ollama (for AI responses):
1. Download from: https://ollama.ai/download
2. Install it
3. Open a new terminal and run: `ollama serve`
4. In another terminal: `ollama pull mistral:7b-instruct`

### Run full API:
```powershell
py simple_api.py
```

---

## 💡 Recommended: Start with Simple Version

**My recommendation:** Start with `simple_api_no_db.py` to see how it works, then upgrade to the full version with PostgreSQL + LLM if you need more advanced features!

---

## 🐛 Troubleshooting

### If packages failed to install:
```powershell
# Make sure venv is active (you should see (venv) in prompt)
pip install fastapi uvicorn pydantic
```

### If port 9000 is busy:
Edit `simple_api_no_db.py` and change:
```python
uvicorn.run(app, host="127.0.0.1", port=9000, log_level="info")
```
to:
```python
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
```

### If you see import errors:
Make sure your virtual environment is activated:
```powershell
.\venv\Scripts\Activate.ps1
```

---

## 🎯 Next Actions

1. **Wait** for package installation to complete
2. **Run** `py simple_api_no_db.py`
3. **Open** browser to `http://127.0.0.1:9000/docs`
4. **Test** the API with sample symptoms
5. **Celebrate!** 🎉

---

**Questions? Let me know which step you're on and I'll help!**
