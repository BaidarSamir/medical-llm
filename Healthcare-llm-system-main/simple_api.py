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
from pydantic import BaseModel, Field
import uvicorn

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
    logger.info("🚀 Starting Symptom Analysis API...")
    
    try:
        # Use fallback model assets to avoid loading issues
        logger.info("📥 Using fallback model configuration...")
        model_assets = {
            "model": None,
            "preprocessor": None,
            "class_names": ["Cardiology", "Neurology", "General Medicine", "Emergency"]
        }
        
        # Create pipeline with fallback model
        pipeline = create_complete_pipeline(model_assets, llm_backend="ollama")
        logger.info("✅ Pipeline initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize pipeline: {e}")
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
        logger.info(f"🔍 Analyzing symptoms: {request.symptom_text[:50]}...")
        
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
        
        logger.info(f"✅ Analysis completed successfully in {result.processing_time:.2f}s")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error during analysis: {e}")
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

def main():
    """Run the API server"""
    logger.info("🚀 Starting Symptom Analysis API Server...")
    
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