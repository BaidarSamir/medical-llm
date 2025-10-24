# ==============================================================================
# FILE: RAG_RETRIEVER.PY
# PURPOSE: RAG Retriever System - Combines Department Classification with Semantic Search
# ==============================================================================

import numpy as np
import tensorflow as tf
import keras_nlp
from typing import Dict, List, Any, Optional, Tuple
import logging
from knowledge_base_postgres import PostgreSQLKnowledgeBase, SymptomDocument

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGRetriever:
    """RAG Retriever that combines department classification with semantic search"""
    
    def __init__(self, roberta_model, preprocessor, class_names, knowledge_base: PostgreSQLKnowledgeBase):
        self.roberta_model = roberta_model
        self.preprocessor = preprocessor
        self.class_names = class_names
        self.knowledge_base = knowledge_base
        
    def classify_department(self, symptom_text: str) -> Tuple[str, float]:
        """Classify the department using the fine-tuned RoBERTa model"""
        try:
            # Check if model is available
            if self.roberta_model is None or self.preprocessor is None:
                logger.warning("⚠️ RoBERTa model not available, using fallback classification")
                # Simple keyword-based fallback classification
                symptom_lower = symptom_text.lower()
                if any(word in symptom_lower for word in ['chest', 'heart', 'cardiac', 'palpitation']):
                    return "Cardiology", 0.7
                elif any(word in symptom_lower for word in ['head', 'brain', 'neurological', 'seizure']):
                    return "Neurology", 0.7
                elif any(word in symptom_lower for word in ['stomach', 'digestive', 'gastro']):
                    return "General Medicine", 0.7
                else:
                    return "General Medicine", 0.5
            
            # Preprocess the input text
            preprocessed_text = self.preprocessor([symptom_text])
            
            # Make prediction
            predictions = self.roberta_model.predict(preprocessed_text, verbose=0)
            
            # Handle different output formats based on number of classes
            if len(self.class_names) == 2:
                # Binary classification
                prediction_score = float(predictions[0][0])
                score_class_1 = prediction_score
                score_class_0 = 1 - score_class_1
                
                predicted_class_index = 1 if score_class_1 > 0.5 else 0
                confidence = score_class_1 if predicted_class_index == 1 else score_class_0
            else:
                # Multi-class classification
                predicted_class_index = int(np.argmax(predictions[0]))
                confidence = float(np.max(predictions[0]))
            
            predicted_department = self.class_names[predicted_class_index]
            
            logger.info(f"🔍 Model prediction: {predicted_department} (confidence: {confidence:.3f})")
            return predicted_department, confidence
            
        except Exception as e:
            logger.error(f"Error in department classification: {e}")
            # Fallback to general medicine if classification fails
            return "General Medicine", 0.5
    
    def retrieve_relevant_documents(self, symptom_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant documents using RAG approach"""
        
        # Step 1: Classify department
        predicted_department, confidence = self.classify_department(symptom_text)
        logger.info(f"Predicted department: {predicted_department} (confidence: {confidence:.3f})")
        
        # Step 2: Search knowledge base with department filter
        search_results = self.knowledge_base.search(
            query=symptom_text,
            top_k=top_k,
            department_filter=predicted_department
        )
        
        # Step 3: If no results with department filter, try without filter
        if not search_results:
            logger.info("No results with department filter, trying broader search...")
            search_results = self.knowledge_base.search(
                query=symptom_text,
                top_k=top_k
            )
        
        # Step 4: Enhance results with classification confidence
        enhanced_results = []
        for result in search_results:
            # The search results contain document data directly, not wrapped in 'document' key
            doc_data = result
            
            # Calculate department match bonus
            department_match = doc_data['department'] == predicted_department
            department_bonus = 0.1 if department_match else 0.0
            
            # Enhanced score combining semantic similarity and department match
            enhanced_score = result['similarity_score'] + department_bonus
            
            enhanced_results.append({
                'symptom': doc_data['symptom'],
                'department': doc_data['department'],
                'reason': doc_data['reason'],
                'causes': doc_data['causes'],
                'consequences': doc_data['consequences'],
                'suggestions': doc_data['suggestions'],
                'content': doc_data['content'],
                'semantic_score': result['similarity_score'],
                'enhanced_score': enhanced_score,
                'department_match': department_match,
                'predicted_department': predicted_department,
                'classification_confidence': confidence,
                'relevance': result['similarity_score']
            })
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x['enhanced_score'], reverse=True)
        
        return enhanced_results
    
    def get_best_match(self, symptom_text: str) -> Optional[Dict[str, Any]]:
        """Get the single best matching document"""
        results = self.retrieve_relevant_documents(symptom_text, top_k=1)
        return results[0] if results else None
    
    def analyze_symptom_severity(self, symptom_text: str) -> Dict[str, Any]:
        """Analyze symptom severity based on retrieved documents"""
        results = self.retrieve_relevant_documents(symptom_text, top_k=3)
        
        if not results:
            return {
                'severity': 'unknown',
                'urgency_level': 'low',
                'recommendation': 'Please provide more detailed symptoms'
            }
        
        # Analyze urgency levels from retrieved documents
        urgency_levels = [result['document'].urgency_level for result in results]
        urgency_scores = {
            'emergency': 4,
            'high': 3,
            'medium': 2,
            'low': 1
        }
        
        # Calculate weighted average urgency
        total_score = 0
        total_weight = 0
        
        for result in results:
            weight = result['enhanced_score']
            urgency_score = urgency_scores.get(result['document'].urgency_level, 1)
            total_score += weight * urgency_score
            total_weight += weight
        
        if total_weight > 0:
            avg_urgency_score = total_score / total_weight
        else:
            avg_urgency_score = 1
        
        # Determine overall severity
        if avg_urgency_score >= 3.5:
            severity = 'critical'
            urgency_level = 'emergency'
        elif avg_urgency_score >= 2.5:
            severity = 'high'
            urgency_level = 'high'
        elif avg_urgency_score >= 1.5:
            severity = 'moderate'
            urgency_level = 'medium'
        else:
            severity = 'low'
            urgency_level = 'low'
        
        # Get recommendations from best match
        best_match = results[0]
        recommendations = best_match['document'].recommendations
        
        return {
            'severity': severity,
            'urgency_level': urgency_level,
            'avg_urgency_score': avg_urgency_score,
            'best_match_department': best_match['document'].department,
            'best_match_title': best_match['document'].title,
            'recommendations': recommendations,
            'classification_confidence': best_match['classification_confidence']
        }

def create_rag_retriever(model_assets: Dict[str, Any], knowledge_base: PostgreSQLKnowledgeBase) -> RAGRetriever:
    """Create a RAG retriever with the given model assets and knowledge base"""
    return RAGRetriever(
        roberta_model=model_assets['model'],
        preprocessor=model_assets['preprocessor'],
        class_names=model_assets['class_names'],
        knowledge_base=knowledge_base
    )

if __name__ == "__main__":
    # Test the RAG retriever
    from knowledge_base import create_knowledge_base
    
    # Create knowledge base
    kb = create_knowledge_base()
    
    # Mock model assets for testing
    # In real usage, these would come from your trained model
    mock_model_assets = {
        'model': None,  # Would be your actual model
        'preprocessor': None,  # Would be your actual preprocessor
        'class_names': np.array(['general_medicine', 'cardiology'])
    }
    
    # Create retriever
    retriever = RAGRetriever(
        roberta_model=mock_model_assets['model'],
        preprocessor=mock_model_assets['preprocessor'],
        class_names=mock_model_assets['class_names'],
        knowledge_base=kb
    )
    
    # Test queries
    test_queries = [
        "I have severe chest pain and shortness of breath",
        "My head hurts really bad and I feel confused",
        "I've been coughing for the past week",
        "My stomach hurts so much I can't move"
    ]
    
    print("\n🔍 Testing RAG Retriever:")
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Test document retrieval
        results = retriever.retrieve_relevant_documents(query, top_k=2)
        print(f"  Retrieved {len(results)} documents:")
        
        for i, result in enumerate(results, 1):
            doc = result['document']
            print(f"    {i}. {doc.title}")
            print(f"       Department: {doc.department}")
            print(f"       Urgency: {doc.urgency_level}")
            print(f"       Enhanced Score: {result['enhanced_score']:.3f}")
        
        # Test severity analysis
        severity_analysis = retriever.analyze_symptom_severity(query)
        print(f"  Severity Analysis:")
        print(f"    Severity: {severity_analysis['severity']}")
        print(f"    Urgency Level: {severity_analysis['urgency_level']}")
        print(f"    Recommendations: {severity_analysis['recommendations']}") 