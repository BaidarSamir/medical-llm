# ==============================================================================
# FILE: LOCAL_LLM.PY
# PURPOSE: Local LLM Integration for Medical Symptom Analysis
# ==============================================================================

import os
import json
import logging
import requests
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMBackend(ABC):
    """Abstract base class for LLM backends"""
    
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response from prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the backend is available"""
        pass

class MockLLMBackend(LLMBackend):
    """Mock LLM backend for testing and development"""
    
    def __init__(self):
        self.responses = {
            "emergency": "🚨 This appears to be a serious medical situation requiring immediate attention. Based on your symptoms, you should seek emergency medical care right away. Please call emergency services or go to the nearest emergency room immediately.",
            "high": "⚠️ Your symptoms indicate a potentially serious condition that requires prompt medical attention. I recommend consulting a healthcare provider as soon as possible.",
            "medium": "Your symptoms suggest a condition that should be evaluated by a healthcare provider. While not immediately life-threatening, it's important to get proper medical assessment.",
            "low": "Your symptoms appear to be mild and may resolve with self-care. However, if symptoms persist or worsen, consider consulting a healthcare provider."
        }
    
    def is_available(self) -> bool:
        return True
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate mock response based on prompt content"""
        time.sleep(0.5)  # Simulate processing time
        
        # Simple keyword-based response selection
        if "emergency" in prompt.lower() or "🚨" in prompt:
            return self.responses["emergency"]
        elif "high" in prompt.lower() or "critical" in prompt.lower():
            return self.responses["high"]
        elif "medium" in prompt.lower():
            return self.responses["medium"]
        else:
            return self.responses["low"]

class LlamaCppBackend(LLMBackend):
    """Llama-cpp-python backend integration"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the llama-cpp model"""
        try:
            from llama_cpp import Llama
            if self.model_path and os.path.exists(self.model_path):
                self.llm = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_threads=4
                )
                logger.info(f"Loaded Llama model from {self.model_path}")
            else:
                logger.warning("Llama model path not found, using mock backend")
                self.llm = None
        except ImportError:
            logger.warning("llama-cpp-python not installed, using mock backend")
            self.llm = None
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using llama-cpp"""
        if not self.is_available():
            raise RuntimeError("Llama model not available")
        
        try:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "\n\n\n"]
            )
            return response['choices'][0]['text'].strip()
        except Exception as e:
            logger.error(f"Error generating response with Llama: {e}")
            raise

class TextGenerationWebUIBackend(LLMBackend):
    """Text-generation-webui API backend"""
    
    def __init__(self, api_url: str = None):
        self.api_url = api_url or os.getenv("TEXT_GENERATION_WEBUI_URL", "http://localhost:7860")
        self.api_endpoint = f"{self.api_url}/api/v1/generate"
    
    def is_available(self) -> bool:
        """Check if the API is available"""
        try:
            response = requests.get(f"{self.api_url}/api/v1/model", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 500, temperature: float = 0.7) -> str:
        """Generate response using text-generation-webui API"""
        if not self.is_available():
            raise RuntimeError("Text-generation-webui API not available")
        
        try:
            payload = {
                "prompt": prompt,
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": 0.9,
                "typical_p": 0.9,
                "repetition_penalty": 1.1,
                "stop": ["</s>", "\n\n\n"]
            }
            
            response = requests.post(self.api_endpoint, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result['results'][0]['text'].strip()
            
        except Exception as e:
            logger.error(f"Error generating response with text-generation-webui: {e}")
            raise

class OllamaBackend(LLMBackend):
    """Ollama backend for local LLM integration"""
    
    def __init__(self, model_name: str = "mistral:7b-instruct", api_url: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            if response.status_code == 200:
                logger.info(f"✅ Connected to Ollama at {self.api_url}")
                # Check if model is available
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                if self.model_name in model_names:
                    logger.info(f"✅ Model {self.model_name} is available")
                else:
                    logger.warning(f"⚠️ Model {self.model_name} not found. Available: {model_names}")
            else:
                logger.warning(f"⚠️ Ollama connection failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Ollama not available: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            response = requests.get(f"{self.api_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> str:
        """Generate response using Ollama with optimized parameters for complete responses"""
        if not self.is_available():
            raise RuntimeError("Ollama not available")
        
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": temperature,
                    "top_k": 40,  # Higher quality sampling
                    "top_p": 0.9,  # Higher quality sampling
                    "repeat_penalty": 1.05,  # Light penalty for quality
                    "num_ctx": 4096,  # Larger context for better understanding
                    "num_thread": 8,  # More threads for CPU processing
                    "num_gpu": 0,  # Force CPU usage since GPU not available
                    "seed": 42,  # Fixed seed for consistency
                    "tfs_z": 0.8,  # Tail free sampling for speed
                    "typical_p": 0.8  # Typical sampling for speed
                },
                "stop": ["\n\n\n", "###", "END", "STOP", "Human:", "Assistant:", "Patient:", "Doctor:"]
            }
            
            response = requests.post(
                f"{self.api_url}/api/generate",
                json=payload,
                timeout=120  # Generous timeout for accuracy over speed
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise RuntimeError(f"Ollama API error: {response.status_code}")
                
        except requests.exceptions.Timeout:
            logger.error("❌ Ollama request timed out - prioritizing accuracy over speed")
            raise RuntimeError("LLM generation timed out - please try again or check system resources")
                
        except Exception as e:
            logger.error(f"Error generating response with Ollama: {e}")
            raise

class LocalLLM:
    """Main LLM interface that manages different backends"""
    
    def __init__(self, backend_type: str = "auto"):
        self.backend_type = backend_type
        self.backend = self._initialize_backend()
    
    def _initialize_backend(self) -> LLMBackend:
        """Initialize the appropriate backend based on configuration"""
        
        if self.backend_type == "mock":
            return MockLLMBackend()
        
        elif self.backend_type == "llama-cpp":
            return LlamaCppBackend()
        
        elif self.backend_type == "text-generation-webui":
            return TextGenerationWebUIBackend()
        
        elif self.backend_type == "ollama":
            return OllamaBackend()
        
        elif self.backend_type == "auto":
            # Try backends in order of preference
            backends = [
                ("text-generation-webui", TextGenerationWebUIBackend),
                ("llama-cpp", LlamaCppBackend),
                ("ollama", OllamaBackend),
                ("mock", MockLLMBackend)
            ]
            
            for name, backend_class in backends:
                try:
                    backend = backend_class()
                    if backend.is_available():
                        logger.info(f"Using {name} backend")
                        return backend
                except Exception as e:
                    logger.warning(f"Failed to initialize {name} backend: {e}")
            
            # Fallback to mock
            logger.warning("No LLM backend available, using mock")
            return MockLLMBackend()
        
        else:
            raise ValueError(f"Unknown backend type: {self.backend_type}")
    
    def generate_response(self, prompt: str, max_tokens: int = 800, temperature: float = 0.7) -> Dict[str, Any]:
        """Generate a response with metadata"""
        
        start_time = time.time()
        
        try:
            response_text = self.backend.generate(prompt, max_tokens, temperature)
            
            generation_time = time.time() - start_time
            
            return {
                "response": response_text,
                "backend": self.backend_type,
                "generation_time": generation_time,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "success": True
            }
            
        except Exception as e:
            generation_time = time.time() - start_time
            
            return {
                "response": f"Error generating response: {str(e)}",
                "backend": self.backend_type,
                "generation_time": generation_time,
                "error": str(e),
                "success": False
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Check the health of the LLM system"""
        return {
            "backend_type": self.backend_type,
            "available": self.backend.is_available(),
            "backend_class": type(self.backend).__name__
        }

def create_local_llm(backend_type: str = "auto") -> LocalLLM:
    """Create a local LLM instance"""
    return LocalLLM(backend_type)

if __name__ == "__main__":
    # Test the local LLM system
    print("🧠 Testing Local LLM System:")
    
    # Test mock backend
    llm = create_local_llm("mock")
    
    test_prompts = [
        "This is an emergency situation with chest pain",
        "I have a mild headache",
        "Severe symptoms requiring immediate attention"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = llm.generate_response(prompt)
        print(f"Response: {result['response']}")
        print(f"Backend: {result['backend']}")
        print(f"Time: {result['generation_time']:.2f}s")
    
    # Test health check
    health = llm.health_check()
    print(f"\nHealth Check: {health}") 
