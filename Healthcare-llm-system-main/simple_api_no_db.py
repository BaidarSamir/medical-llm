#!/usr/bin/env python3
"""
Simple API Server - No PostgreSQL Required
Uses file-based knowledge base instead of database
"""

import os
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Medical Symptom Analysis API (Simple)",
    description="Symptom analysis using file-based knowledge base",
    version="1.0.0"
)

# Import the simple knowledge base
try:
    from simple_knowledge_base import SimpleKnowledgeBase
    knowledge_base = SimpleKnowledgeBase(kb_folder="KB_")
    logger.info(f"✅ Loaded {len(knowledge_base.documents)} medical documents")
except Exception as e:
    logger.error(f"❌ Failed to load knowledge base: {e}")
    knowledge_base = None

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

@app.get("/")
async def root():
    """API information endpoint"""
    return {
        "name": "Medical Symptom Analysis API (Simple)",
        "version": "1.0.0",
        "status": "operational",
        "mode": "simple_file_based",
        "documents_loaded": len(knowledge_base.documents) if knowledge_base else 0,
        "endpoints": {
            "POST /analyze": "Analyze symptoms and get recommendations",
            "GET /health": "Health check",
            "GET /departments": "List available departments",
            "GET /docs": "Interactive API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "knowledge_base": "loaded" if knowledge_base and knowledge_base.documents else "not loaded",
        "documents": len(knowledge_base.documents) if knowledge_base else 0
    }

@app.get("/departments")
async def get_departments():
    """Get list of available medical departments"""
    if not knowledge_base or not knowledge_base.documents:
        return {"departments": ["Cardiology", "General Medicine"]}
    
    departments = list(set([doc.department for doc in knowledge_base.documents]))
    return {"departments": departments}

@app.post("/analyze", response_model=SymptomResponse)
async def analyze_symptoms(request: SymptomRequest):
    """
    Analyze patient symptoms and provide medical guidance
    
    Args:
        request: SymptomRequest containing symptom description
        
    Returns:
        SymptomResponse with department and description
    """
    try:
        symptom_text = request.symptom_text.lower()
        
        # Check if knowledge base is available
        if not knowledge_base or not knowledge_base.documents:
            return SymptomResponse(
                department="General Medicine",
                description="I understand you're experiencing some symptoms. While I don't have access to the full medical knowledge base at the moment, I recommend consulting with a healthcare professional who can properly evaluate your condition and provide appropriate guidance."
            )
        
        # Search for matching symptoms
        best_match = None
        best_score = 0
        
        for doc in knowledge_base.documents:
            score = 0
            # Check main symptom
            if doc.symptom.lower() in symptom_text:
                score += 10
            
            # Check aliases
            for alias in doc.aliases:
                if alias.lower() in symptom_text:
                    score += 5
            
            # Check keywords from causes
            for cause in doc.causes[:3]:  # Top 3 causes
                cause_words = cause.lower().split()
                for word in cause_words:
                    if len(word) > 3 and word in symptom_text:
                        score += 1
            
            if score > best_score:
                best_score = score
                best_match = doc
        
        # If we found a match, generate response
        if best_match and best_score >= 3:
            department = best_match.department
            
            # Build description
            description = f"Based on your symptoms, this appears to be related to {best_match.symptom}. "
            description += f"{best_match.reason} "
            
            # Add potential causes
            if best_match.causes:
                description += f"This could be caused by: {', '.join(best_match.causes[:3])}. "
            
            # Add suggestions
            if best_match.suggestions:
                description += f"I recommend: {' '.join(best_match.suggestions[:3])}"
            
            return SymptomResponse(
                department=department,
                description=description
            )
        else:
            # Generic response if no good match
            return SymptomResponse(
                department="General Medicine",
                description="I understand you're experiencing symptoms. Based on the information provided, I recommend consulting with a healthcare professional for a proper evaluation. They can assess your condition thoroughly and provide appropriate guidance and treatment options."
            )
        
    except Exception as e:
        logger.error(f"❌ Error analyzing symptoms: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")

if __name__ == "__main__":
    logger.info("🚀 Starting Simple Medical Symptom Analysis API...")
    logger.info("📚 Using file-based knowledge base (no PostgreSQL required)")
    logger.info("🌐 Server will be available at: http://127.0.0.1:9000")
    logger.info("📖 API documentation at: http://127.0.0.1:9000/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=9000, log_level="info")
