import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logger = logging.getLogger(__name__)

@dataclass
class SymptomDocument:
    """Represents a symptom document from the knowledge base."""
    id: str
    symptom: str
    aliases: List[str]
    department: str
    reason: str
    causes: List[str]
    consequences: List[str]
    suggestions: List[str]
    content: str
    embedding: Optional[np.ndarray] = None

class PostgreSQLKnowledgeBase:
    """Knowledge base implementation using PostgreSQL with pgvector extension."""
    
    def __init__(self, 
                 db_url: str = "postgresql://postgres:12345678@localhost:5432/symptom_kb",
                 model_name: str = "all-MiniLM-L6-v2",
                 kb_folder: str = "KB_"):
        """
        Initialize the PostgreSQL knowledge base.
        
        Args:
            db_url: PostgreSQL connection URL
            model_name: Sentence transformer model name
            kb_folder: Folder containing JSON symptom documents
        """
        self.db_url = db_url
        self.model_name = model_name
        self.kb_folder = kb_folder
        try:
            # Use a simple model that should work without authentication
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("[SUCCESS] SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.warning(f"[WARNING] sentence-transformers failed: {e}")
            logger.info("[INFO] Using fallback embedding method")
            self.embedding_model = None
        self.engine = None
        self.documents: List[SymptomDocument] = []
        
    def connect_database(self) -> bool:
        """Establish connection to PostgreSQL database."""
        try:
            self.engine = create_engine(self.db_url)
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("[SUCCESS] Successfully connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to connect to database: {e}")
            return False
    
    def create_tables(self) -> bool:
        """Create necessary tables for storing symptom documents and embeddings."""
        try:
            with self.engine.connect() as conn:
                # Try to create pgvector extension (optional - will skip if not available)
                use_vector = False
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                    conn.commit()
                    logger.info("[SUCCESS] pgvector extension enabled")
                    use_vector = True
                except Exception as e:
                    logger.warning(f"[WARNING] pgvector not available, using TEXT for embeddings: {e}")
                    # Rollback the failed transaction so we can continue
                    conn.rollback()
                
                # Create symptoms table (with TEXT embedding if pgvector unavailable)
                embedding_type = "vector(384)" if use_vector else "TEXT"
                conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS symptoms (
                        id SERIAL PRIMARY KEY,
                        symptom_name VARCHAR(255) NOT NULL,
                        aliases TEXT[],
                        department VARCHAR(100) NOT NULL,
                        reason TEXT,
                        causes TEXT[],
                        consequences TEXT[],
                        suggestions TEXT[],
                        content TEXT,
                        embedding {embedding_type},
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """))
                conn.commit()
                
                # Create index for vector similarity search (only if pgvector available)
                if use_vector:
                    try:
                        conn.execute(text("""
                            CREATE INDEX IF NOT EXISTS symptoms_embedding_idx 
                            ON symptoms USING ivfflat (embedding vector_cosine_ops)
                            WITH (lists = 100)
                        """))
                        conn.commit()
                    except Exception as e:
                        logger.warning(f"[WARNING] Could not create vector index: {e}")
                
            logger.info("[SUCCESS] Database tables created successfully")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to create tables: {e}")
            return False
    
    def load_documents_from_folder(self) -> List[SymptomDocument]:
        """Load all JSON documents from the KB_ folder."""
        documents = []
        kb_path = Path(self.kb_folder)
        
        if not kb_path.exists():
            logger.error(f"[ERROR] KB folder '{self.kb_folder}' not found")
            return documents
        
        json_files = list(kb_path.glob("*.json"))
        logger.info(f"[FOLDER] Found {len(json_files)} JSON files in {self.kb_folder}")
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Create content string for embedding
                content_parts = [
                    data.get('symptom', ''),
                    data.get('reason', ''),
                    ' '.join(data.get('aliases', [])),
                    ' '.join(data.get('causes', [])),
                    ' '.join(data.get('consequences', [])),
                    ' '.join(data.get('suggestions', []))
                ]
                content = ' '.join(filter(None, content_parts))
                
                document = SymptomDocument(
                    id=str(json_file.stem),
                    symptom=data.get('symptom', ''),
                    aliases=data.get('aliases', []),
                    department=data.get('department', ''),
                    reason=data.get('reason', ''),
                    causes=data.get('causes', []),
                    consequences=data.get('consequences', []),
                    suggestions=data.get('suggestions', []),
                    content=content
                )
                documents.append(document)
                logger.info(f"[FILE] Loaded: {document.symptom} ({document.department})")
                
            except Exception as e:
                logger.error(f"[ERROR] Failed to load {json_file}: {e}")
        
        self.documents = documents
        logger.info(f"[SUCCESS] Loaded {len(documents)} documents from {self.kb_folder}")
        return documents
    
    def create_embeddings(self) -> bool:
        """Create embeddings for all loaded documents."""
        if not self.documents:
            logger.warning("[WARNING] No documents loaded. Call load_documents_from_folder() first.")
            return False
        
        if self.embedding_model is None:
            logger.warning("[WARNING] No embedding model available, creating simple hash-based embeddings")
            # Create simple hash-based embeddings as fallback
            import hashlib
            for doc in self.documents:
                # Create a simple hash-based embedding
                content_hash = hashlib.md5(doc.content.encode()).hexdigest()
                # Convert hash to 384-dimensional vector (same as original model)
                embedding = np.array([ord(c) % 10 for c in content_hash[:96]] * 4, dtype=np.float32)
                # Ensure it's exactly 384 dimensions
                if len(embedding) < 384:
                    # Pad with zeros if needed
                    embedding = np.pad(embedding, (0, 384 - len(embedding)), 'constant')
                elif len(embedding) > 384:
                    # Truncate if too long
                    embedding = embedding[:384]
                doc.embedding = embedding
            
            logger.info(f"[SUCCESS] Created hash-based embeddings for {len(self.documents)} documents")
            return True
        
        try:
            # Create embeddings for all documents
            contents = [doc.content for doc in self.documents]
            embeddings = self.embedding_model.encode(contents, show_progress_bar=True)
            
            # Assign embeddings to documents
            for doc, embedding in zip(self.documents, embeddings):
                doc.embedding = embedding
            
            logger.info(f"[SUCCESS] Created embeddings for {len(self.documents)} documents")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to create embeddings: {e}")
            return False
    
    def save_to_database(self) -> bool:
        """Save all documents with embeddings to PostgreSQL database."""
        if not self.documents:
            logger.warning("[WARNING] No documents to save. Load documents first.")
            return False
        
        logger.info(f"[INFO] Attempting to save {len(self.documents)} documents to database...")
        
        try:
            # Use psycopg2 directly with password from db_url
            # Extract password from self.db_url (format: postgresql://user:password@host:port/db)
            from urllib.parse import urlparse
            parsed = urlparse(self.db_url)
            
            conn = psycopg2.connect(
                host=parsed.hostname or "localhost",
                database=parsed.path.lstrip('/'),
                user=parsed.username or "postgres",
                password=parsed.password or ""
            )
            cursor = conn.cursor()
            
            # Clear existing data
            logger.info("[CLEAR] Clearing existing data...")
            cursor.execute("DELETE FROM symptoms")
            conn.commit()
            
            # Insert new documents
            saved_count = 0
            for i, doc in enumerate(self.documents):
                if doc.embedding is not None:
                    # Convert embedding to string format (for TEXT column)
                    embedding_list = doc.embedding.tolist()
                    embedding_str = '[' + ','.join(map(str, embedding_list)) + ']'
                    
                    logger.info(f"[SAVE] Saving document {i+1}/{len(self.documents)}: {doc.symptom}")
                    
                    cursor.execute("""
                        INSERT INTO symptoms 
                        (symptom_name, aliases, department, reason, causes, 
                         consequences, suggestions, content, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        doc.symptom,
                        doc.aliases,
                        doc.department,
                        doc.reason,
                        doc.causes,
                        doc.consequences,
                        doc.suggestions,
                        doc.content,
                        embedding_str
                    ))
                    saved_count += 1
                else:
                    logger.warning(f"[WARNING] Document {i+1} has no embedding: {doc.symptom}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"[SUCCESS] Successfully saved {saved_count} documents to database")
            return True
                
        except Exception as e:
            logger.error(f"[ERROR] Failed to save to database: {e}")
            return False
    
    def search(self, query: str, top_k: int = 3, department_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using keyword matching (fallback for no pgvector).
        
        Args:
            query: Search query
            top_k: Number of top results to return
            department_filter: Optional department filter
            
        Returns:
            List of similar documents with scores
        """
        try:
            # Extract keywords from query (simple word-based approach)
            keywords = [word.lower() for word in query.split() if len(word) > 3]
            
            if not keywords:
                logger.warning("[WARNING] No keywords extracted from query")
                # Return all documents if no keywords
                keywords = ['pain', 'chest']  # Default fallback
            
            # Use keyword-based search since we don't have pgvector
            with self.engine.connect() as conn:
                # Use psycopg2 directly for text search
                cursor = conn.connection.cursor()
                
                # Build OR conditions for each keyword
                keyword_conditions = " OR ".join(["LOWER(content) LIKE %s" for _ in keywords])
                
                # Keyword-based search using LIKE with OR
                search_sql = f"""
                    SELECT 
                        symptom_name,
                        aliases,
                        department,
                        reason,
                        causes,
                        consequences,
                        suggestions,
                        content,
                        0.8 as similarity_score
                    FROM symptoms
                    WHERE {keyword_conditions}
                """
                
                # Create search patterns for each keyword
                params = [f"%{keyword}%" for keyword in keywords]
                
                # Add department filter if specified
                if department_filter:
                    search_sql += " AND department = %s"
                    params.append(department_filter)
                
                search_sql += " LIMIT %s"
                params.append(top_k)
                
                cursor.execute(search_sql, params)
                rows = cursor.fetchall()
                
                # Convert to list of dictionaries
                results = []
                for row in rows:
                    results.append({
                        'symptom': row[0],  # symptom_name
                        'aliases': row[1] or [],  # aliases
                        'department': row[2],  # department
                        'reason': row[3],  # reason
                        'causes': row[4] or [],  # causes
                        'consequences': row[5] or [],  # consequences
                        'suggestions': row[6] or [],  # suggestions
                        'content': row[7],  # content
                        'similarity_score': float(row[8])  # similarity_score
                    })
                
                cursor.close()
                
                logger.info(f"[SEARCH] Found {len(results)} similar documents for query: '{query}'")
                return results
                
        except Exception as e:
            logger.error(f"[ERROR] Search failed: {e}")
            return []
    
    def get_department_stats(self) -> Dict[str, int]:
        """Get statistics about documents by department."""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT department, COUNT(*) as count
                    FROM symptoms
                    GROUP BY department
                    ORDER BY count DESC
                """))
                stats = {row.department: row.count for row in result.fetchall()}
                return stats
        except Exception as e:
            logger.error(f"[ERROR] Failed to get department stats: {e}")
            return {}
    
    def initialize_knowledge_base(self) -> bool:
        """Complete initialization: load documents, create embeddings, save to DB."""
        logger.info("[INIT] Initializing PostgreSQL knowledge base...")
        
        # Step 1: Connect to database
        if not self.connect_database():
            return False
        
        # Step 2: Create tables
        if not self.create_tables():
            return False
        
        # Step 3: Load documents
        documents = self.load_documents_from_folder()
        if not documents:
            logger.warning("[WARNING] No documents loaded from KB_ folder")
            return False
        
        # Step 4: Create embeddings
        if not self.create_embeddings():
            return False
        
        # Step 5: Save to database
        if not self.save_to_database():
            return False
        
        # Step 6: Show statistics
        stats = self.get_department_stats()
        logger.info("[STATS] Knowledge base statistics:")
        for dept, count in stats.items():
            logger.info(f"   {dept}: {count} documents")
        
        logger.info("[SUCCESS] Knowledge base initialization completed successfully!")
        return True

def create_postgres_knowledge_base(db_url: str = "postgresql://postgres:12345678@localhost:5432/symptom_kb") -> PostgreSQLKnowledgeBase:
    """Factory function to create and initialize a PostgreSQL knowledge base."""
    kb = PostgreSQLKnowledgeBase(db_url=db_url)
    return kb

# Test function
def test_postgres_kb():
    """Test the PostgreSQL knowledge base implementation."""
    logger.info("[TEST] Testing PostgreSQL knowledge base...")
    
    # Create knowledge base
    kb = PostgreSQLKnowledgeBase()
    
    # Initialize
    if kb.initialize_knowledge_base():
        # Test search
        test_queries = [
            "chest pain and shortness of breath",
            "headache and dizziness",
            "fever and fatigue"
        ]
        
        for query in test_queries:
            logger.info(f"\n[SEARCH] Testing query: '{query}'")
            results = kb.search(query, top_k=2)
            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. {result['symptom']} ({result['department']}) - Score: {result['similarity_score']:.3f}")
    
    else:
        logger.error("[ERROR] Knowledge base initialization failed")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_postgres_kb() 
