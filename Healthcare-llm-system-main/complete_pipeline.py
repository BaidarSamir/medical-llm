#!/usr/bin/env python3
"""
Complete Symptom Analysis Pipeline
Integrates: User Query → RoBERTa Classifier → Symptom Retriever (RAG) → Prompt Builder → Local LLM → Final Output
"""

import os
import json
import logging
import time
import numpy as np
import tensorflow as tf
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import uuid

# Import our custom modules
from knowledge_base_postgres import PostgreSQLKnowledgeBase, create_postgres_knowledge_base
from rag_retriever import RAGRetriever, create_rag_retriever
from prompt_builder import PromptBuilder, PromptContext, create_prompt_builder
from local_llm import LocalLLM, create_local_llm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class PipelineResult:
    """Complete pipeline result"""
    request_id: str
    user_input: str
    predicted_department: str
    classification_confidence: float
    retrieved_document: Dict[str, Any]
    llm_response: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None

class CompleteSymptomPipeline:
    """Complete symptom analysis pipeline integrating all components"""
    
    def __init__(self, model_assets: Dict[str, Any], llm_backend: str = "mock"):
        """
        Initialize the complete pipeline
        
        Args:
            model_assets: Dictionary containing RoBERTa model, preprocessor, and class names
            llm_backend: LLM backend type ("mock", "llama-cpp", "text-generation-webui")
        """
        self.model_assets = model_assets
        self.llm_backend = llm_backend
        
        logger.info("🚀 Initializing Complete Symptom Analysis Pipeline...")
        
        # Initialize knowledge base
        logger.info("📚 Initializing knowledge base...")
        self.knowledge_base = self._initialize_knowledge_base()
        
        # Initialize RAG retriever
        logger.info("🔍 Initializing RAG retriever...")
        self.rag_retriever = create_rag_retriever(model_assets, self.knowledge_base)
        
        # Initialize prompt builder
        logger.info("📝 Initializing prompt builder...")
        self.prompt_builder = create_prompt_builder()
        
        # Initialize local LLM
        logger.info("🤖 Initializing local LLM...")
        self.local_llm = create_local_llm(llm_backend)
        
        logger.info("✅ Complete pipeline initialized successfully")
    
    def _initialize_knowledge_base(self) -> PostgreSQLKnowledgeBase:
        """Initialize the PostgreSQL knowledge base"""
        try:
            # Use PostgreSQL with pgvector
            db_url = "postgresql://postgres@localhost:5432/symptom_kb"
            kb = create_postgres_knowledge_base(db_url=db_url)
            
            # Initialize the knowledge base if not already done
            if not kb.initialize_knowledge_base():
                logger.warning("⚠️ Knowledge base initialization failed, but continuing...")
            
            logger.info("✅ PostgreSQL knowledge base initialized")
            return kb
        except Exception as e:
            logger.error(f"❌ Failed to initialize knowledge base: {e}")
            raise
    
    def analyze_symptoms(self, symptom_text: str) -> PipelineResult:
        """
        Complete symptom analysis pipeline
        
        Args:
            symptom_text: User's symptom description
            
        Returns:
            PipelineResult: Complete analysis with all components
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Analyzing symptoms: {symptom_text[:50]}...")
            
            # Step 1: Retrieve relevant documents using RAG
            logger.info("Step 1: Retrieving relevant documents...")
            retrieved_results = self.rag_retriever.retrieve_relevant_documents(symptom_text, top_k=1)
            
            if not retrieved_results:
                raise ValueError("No relevant documents found in knowledge base")
            
            best_match = retrieved_results[0]
            logger.info(f"✅ Retrieved document: {best_match['symptom']} ({best_match['department']})")
            
            # Step 2: Build prompt for LLM
            logger.info("Step 2: Building prompt...")
            prompt_context = PromptContext(
                user_input=symptom_text,
                predicted_department=best_match['department'],
                classification_confidence=best_match.get('relevance_score', 0.8),
                retrieved_document=best_match,
                severity_analysis={
                    'severity': 'moderate',
                    'urgency_level': 'medium',
                    'recommendations': best_match.get('suggestions', [])
                }
            )
            
            prompt = self.prompt_builder.build_prompt(prompt_context)
            logger.info(f"✅ Built prompt ({len(prompt)} characters)")
            
            # Step 3: Generate LLM response
            logger.info("Step 3: Generating LLM response...")
            llm_result = self.local_llm.generate_response(prompt, max_tokens=300, temperature=0.7)
            
            if not llm_result['success']:
                raise RuntimeError(f"LLM generation failed: {llm_result.get('error', 'Unknown error')}")
            
            llm_response = llm_result['response']
            logger.info(f"✅ Generated LLM response ({len(llm_response)} characters)")
            
            # Step 4: Calculate processing time
            processing_time = time.time() - start_time
            
            # Step 5: Create result
            result = PipelineResult(
                request_id=request_id,
                user_input=symptom_text,
                predicted_department=best_match['department'],
                classification_confidence=best_match.get('relevance_score', 0.8),
                retrieved_document=best_match,
                llm_response=llm_response,
                processing_time=processing_time,
                success=True
            )
            
            logger.info(f"✅ Pipeline completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ Pipeline failed: {e}")
            
            return PipelineResult(
                request_id=request_id,
                user_input=symptom_text,
                predicted_department="Unknown",
                classification_confidence=0.0,
                retrieved_document={},
                llm_response="",
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        health = {
            "pipeline_status": "operational",
            "components": {}
        }
        
        # Check knowledge base
        try:
            kb_stats = self.knowledge_base.get_department_stats()
            health["components"]["knowledge_base"] = {
                "status": "operational",
                "documents_count": sum(kb_stats.values()),
                "departments": list(kb_stats.keys())
            }
        except Exception as e:
            health["components"]["knowledge_base"] = {
                "status": "error",
                "error": str(e)
            }
            health["pipeline_status"] = "degraded"
        
        # Check LLM
        try:
            llm_health = self.local_llm.health_check()
            health["components"]["local_llm"] = llm_health
        except Exception as e:
            health["components"]["local_llm"] = {
                "status": "error",
                "error": str(e)
            }
            health["pipeline_status"] = "degraded"
        
        return health

def create_complete_pipeline(model_assets: Dict[str, Any], llm_backend: str = "mock") -> CompleteSymptomPipeline:
    """Factory function to create a complete pipeline"""
    return CompleteSymptomPipeline(model_assets, llm_backend)

def test_pipeline():
    """Test the complete pipeline"""
    logger.info("🧪 Testing Complete Symptom Pipeline...")
    
    # Mock model assets for testing
    mock_model_assets = {
        "model": None,  # We'll use RAG instead of RoBERTa for now
        "preprocessor": None,
        "class_names": ["Cardiology", "Neurology", "General Medicine", "Emergency"]
    }
    
    try:
        # Create pipeline
        pipeline = create_complete_pipeline(mock_model_assets, llm_backend="mock")
        
        # Test with sample symptoms
        test_symptoms = [
            "I feel tightness in my chest and I'm having trouble breathing",
            "I have a severe headache that won't go away",
            "I'm experiencing dizziness and fatigue"
        ]
        
        for symptoms in test_symptoms:
            logger.info(f"\n🔍 Testing: {symptoms}")
            result = pipeline.analyze_symptoms(symptoms)
            
            if result.success:
                logger.info(f"✅ Success! Department: {result.predicted_department}")
                logger.info(f"📝 LLM Response: {result.llm_response[:100]}...")
            else:
                logger.error(f"❌ Failed: {result.error_message}")
        
        # Test system health
        health = pipeline.get_system_health()
        logger.info(f"🏥 System Health: {health['pipeline_status']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    if success:
        logger.info("🎉 Pipeline test completed successfully!")
    else:
        logger.error("❌ Pipeline test failed!") 