#!/usr/bin/env python3
"""
Simple Knowledge Base for Symptom Analysis
Uses keyword matching instead of vector embeddings for simplicity
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SymptomDocument:
    """Represents a symptom document from the knowledge base."""
    symptom: str
    aliases: List[str]
    department: str
    reason: str
    causes: List[str]
    consequences: List[str]
    suggestions: List[str]
    content: str

class SimpleKnowledgeBase:
    """Simple knowledge base using keyword matching."""
    
    def __init__(self, kb_folder: str = "KB_"):
        """
        Initialize the simple knowledge base.
        
        Args:
            kb_folder: Folder containing JSON symptom documents
        """
        self.kb_folder = kb_folder
        self.documents: List[SymptomDocument] = []
        self._load_documents()
    
    def _load_documents(self):
        """Load all documents from the KB_ folder."""
        kb_path = Path(self.kb_folder)
        if not kb_path.exists():
            logger.error(f"❌ KB_ folder not found: {self.kb_folder}")
            return
        
        json_files = list(kb_path.glob("*.json"))
        if not json_files:
            logger.error(f"❌ No JSON files found in {self.kb_folder}")
            return
        
        logger.info(f"📁 Found {len(json_files)} JSON files in {self.kb_folder}")
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Create content string for searching
                content_parts = [
                    data.get('symptom', ''),
                    ' '.join(data.get('aliases', [])),
                    data.get('reason', ''),
                    ' '.join(data.get('causes', [])),
                    ' '.join(data.get('consequences', [])),
                    ' '.join(data.get('suggestions', []))
                ]
                content = ' '.join(content_parts).lower()
                
                doc = SymptomDocument(
                    symptom=data.get('symptom', ''),
                    aliases=data.get('aliases', []),
                    department=data.get('department', ''),
                    reason=data.get('reason', ''),
                    causes=data.get('causes', []),
                    consequences=data.get('consequences', []),
                    suggestions=data.get('suggestions', []),
                    content=content
                )
                
                self.documents.append(doc)
                logger.info(f"📄 Loaded: {doc.symptom} ({doc.department})")
                
            except Exception as e:
                logger.error(f"❌ Failed to load {file_path}: {e}")
        
        logger.info(f"✅ Loaded {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 3, department_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using keyword matching.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            department_filter: Optional department filter
            
        Returns:
            List of similar documents with scores
        """
        query_lower = query.lower()
        query_words = set(re.findall(r'\w+', query_lower))
        
        results = []
        
        for doc in self.documents:
            # Skip if department filter is specified and doesn't match
            if department_filter and doc.department.lower() != department_filter.lower():
                continue
            
            # Calculate similarity score based on word overlap
            doc_words = set(re.findall(r'\w+', doc.content))
            
            if not query_words or not doc_words:
                continue
            
            # Calculate Jaccard similarity
            intersection = len(query_words.intersection(doc_words))
            union = len(query_words.union(doc_words))
            similarity = intersection / union if union > 0 else 0
            
            # Bonus for exact symptom match
            if doc.symptom.lower() in query_lower:
                similarity += 0.3
            
            # Bonus for alias matches
            for alias in doc.aliases:
                if alias.lower() in query_lower:
                    similarity += 0.2
                    break
            
            if similarity > 0:
                results.append({
                    'symptom': doc.symptom,
                    'aliases': doc.aliases,
                    'department': doc.department,
                    'reason': doc.reason,
                    'causes': doc.causes,
                    'consequences': doc.consequences,
                    'suggestions': doc.suggestions,
                    'content': doc.content,
                    'relevance_score': min(similarity, 1.0)
                })
        
        # Sort by relevance score and return top_k
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results[:top_k]
    
    def get_department_stats(self) -> Dict[str, int]:
        """Get statistics about documents by department."""
        stats = {}
        for doc in self.documents:
            dept = doc.department
            stats[dept] = stats.get(dept, 0) + 1
        return stats
    
    def get_all_departments(self) -> List[str]:
        """Get list of all departments."""
        return list(set(doc.department for doc in self.documents))

def create_simple_knowledge_base(kb_folder: str = "KB_") -> SimpleKnowledgeBase:
    """Factory function to create a simple knowledge base."""
    return SimpleKnowledgeBase(kb_folder)

def test_simple_kb():
    """Test the simple knowledge base."""
    logger.info("🧪 Testing Simple Knowledge Base...")
    
    try:
        kb = create_simple_knowledge_base()
        
        # Test search
        test_queries = [
            "I feel tightness in my chest and I'm having trouble breathing",
            "I have a severe headache that won't go away",
            "I'm experiencing dizziness and fatigue"
        ]
        
        for query in test_queries:
            logger.info(f"\n🔍 Testing query: {query[:50]}...")
            results = kb.search(query, top_k=2)
            
            if results:
                logger.info(f"✅ Found {len(results)} results")
                for i, result in enumerate(results):
                    logger.info(f"  {i+1}. {result['symptom']} ({result['department']}) - Score: {result['relevance_score']:.3f}")
            else:
                logger.warning("⚠️ No results found")
        
        # Test department stats
        stats = kb.get_department_stats()
        logger.info(f"\n📊 Department statistics: {stats}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Simple KB test failed: {e}")
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    success = test_simple_kb()
    if success:
        logger.info("🎉 Simple KB test completed successfully!")
    else:
        logger.error("❌ Simple KB test failed!") 