# Ollama Setup Guide for Local LLM Integration

## 🚀 Quick Setup

### 1. Install Ollama
Visit: https://ollama.ai/download

**Windows:**
- Download and run the Windows installer
- Follow the installation wizard

**Linux/macOS:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Start Ollama
```bash
ollama serve
```

### 3. Download Mistral 7B Instruct
```bash
ollama pull mistral:7b-instruct
```

**Alternative Models (if needed):**
- **Mistral 7B (Base):** `ollama pull mistral`
- **Llama 2 (Good Quality):** `ollama pull llama2`
- **Llama 2 Medical (Specialized):** `ollama pull llama2:13b`

### 4. Test the Model
```bash
ollama run mistral:7b-instruct "Hello, how are you?"
```

## 🔧 Integration with Your Pipeline

Your pipeline is now configured to use Ollama with **Mistral 7B Instruct** automatically. The system will:

1. **Try to connect to Ollama** at `http://localhost:11434`
2. **Use Mistral 7B Instruct** model (`mistral:7b-instruct`)
3. **Generate natural responses** based on your medical prompts
4. **Fall back to mock** if Ollama is not available

## 📝 Expected Response Format

With Ollama + improved prompts, you should get responses like:

```json
{
  "department": "Cardiology",
  "description": "It sounds like you're experiencing symptoms related to your heart, such as chest tightness and difficulty breathing. These could be signs that your heart isn't getting enough oxygen. This can happen due to things like high blood pressure or blocked arteries. It's important to take these symptoms seriously, as they could lead to a heart attack if not addressed. I recommend getting an ECG and seeing a cardiologist as soon as possible. Also, try to stay calm and avoid any physical or emotional stress."
}
```

## 🎯 Next Steps

1. **Install Ollama** using the steps above
2. **Download a model** (mistral recommended)
3. **Restart your API server**
4. **Test with your symptoms**

The system will automatically detect and use Ollama for natural, detailed responses! 