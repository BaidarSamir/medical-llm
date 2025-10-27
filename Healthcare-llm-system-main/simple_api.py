#!/usr/bin/env python3
"""
Simple API for Symptom Analysis Pipeline
Returns JSON in the format: {"department": "...", "description": "..."}
"""

import os
import json
import logging
import time
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from fastapi.staticfiles import StaticFiles

# Import our pipeline
from complete_pipeline import CompleteSymptomPipeline, create_complete_pipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Symptom Analysis API",
    description="Complete symptom analysis pipeline with RAG and local LLM",
    version="1.0.0"
)

# Add CORS middleware for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

class SymptomRequest(BaseModel):
    symptom_text: str = Field(
        ..., 
        min_length=5, 
        max_length=500,
        title="Symptom Text",
        description="Description of the patient's symptoms",
        example="I feel tightness in my chest and I'm having trouble breathing"
    )

class SymptomResponse(BaseModel):
    department: str = Field(..., description="Recommended medical department")
    description: str = Field(..., description="Detailed explanation and recommendations")

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    global pipeline
    logger.info("[STARTUP] Starting Symptom Analysis API...")
    
    try:
        # Try to load Medical RoBERTa model
        logger.info("[STARTUP] Attempting to load Medical RoBERTa model...")
        try:
            from medical_roberta_loader import load_medical_roberta_model
            model_assets = load_medical_roberta_model(
                class_names=["Cardiology", "Neurology", "General Medicine", "Emergency"]
            )
            if model_assets["model"] is not None:
                logger.info("[SUCCESS] Medical RoBERTa loaded successfully!")
            else:
                logger.info("[WARNING] Using fallback classifier (Medical RoBERTa not available)")
        except Exception as e:
            logger.warning(f"[WARNING] Could not load Medical RoBERTa: {e}")
            logger.info("[STARTUP] Using fallback model configuration...")
            model_assets = {
                "model": None,
                "preprocessor": None,
                "class_names": ["Cardiology", "Neurology", "General Medicine", "Emergency"]
            }
        
        # Create pipeline
        pipeline = create_complete_pipeline(model_assets, llm_backend="ollama")
        logger.info("[SUCCESS] Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize pipeline: {e}")
        raise

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Symptom Analysis API",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        health = pipeline.get_system_health()
        return {
            "status": "healthy",
            "pipeline_status": health["pipeline_status"],
            "components": health["components"]
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")

@app.post("/analyze", response_model=SymptomResponse)
async def analyze_symptoms(request: SymptomRequest):
    """
    Analyze symptoms and return department recommendation with description
    
    This endpoint implements the complete pipeline:
    1. Accepts symptom text from user
    2. Uses RAG to retrieve relevant symptom info from knowledge base
    3. Feeds it into a local LLM to generate a single description
    4. Returns JSON with department and description
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        logger.info(f"[ANALYZE] Analyzing symptoms: {request.symptom_text[:50]}...")
        
        # Run the complete pipeline
        result = pipeline.analyze_symptoms(request.symptom_text)
        
        if not result.success:
            raise HTTPException(
                status_code=500, 
                detail=f"Analysis failed: {result.error_message}"
            )
        
        # Return the exact JSON format requested
        response = SymptomResponse(
            department=result.predicted_department,
            description=result.llm_response
        )
        
        logger.info(f"[SUCCESS] Analysis completed successfully in {result.processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[ERROR] Unexpected error during analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/departments")
async def get_departments():
    """Get available medical departments"""
    return {
        "departments": [
            "Cardiology",
            "Neurology", 
            "General Medicine",
            "Emergency",
            "Orthopedics",
            "Dermatology",
            "Gastroenterology"
        ]
    }

# Serve the frontend if built to webui/dist
if os.path.isdir(os.path.join(os.path.dirname(__file__), 'webui', 'dist')):
    app.mount('/ui', StaticFiles(directory=os.path.join(os.path.dirname(__file__), 'webui', 'dist'), html=True), name='webui')
else:
    # Provide a helpful message at /ui when frontend not built
    @app.get('/ui')
    def web_ui_not_built():
        return {"status":"no-frontend","message":"Frontend not built. Run 'cd webui && npm install && npm run build' and place the contents in webui/dist."}

def main():
    """Run the API server"""
    logger.info("[MAIN] Starting Symptom Analysis API Server...")
    
    # Run with uvicorn
    uvicorn.run(
        "simple_api:app",
        host="127.0.0.1",
        port=9001,  # Changed to port 9001
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()