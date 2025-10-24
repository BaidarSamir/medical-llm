"""
Initialize Knowledge Base - Setup Script
This script will load medical documents into PostgreSQL database
"""

import os
import sys
from pathlib import Path

print("=" * 60)
print("🚀 Medical LLM - Knowledge Base Initialization")
print("=" * 60)
print()

# Get PostgreSQL password
password = input("Enter your PostgreSQL password for user 'postgres': ")
print()

# Set environment variable for database connection
db_url = f"postgresql://postgres:{password}@localhost:5432/symptom_kb"
os.environ['DATABASE_URL'] = db_url

# Now import and run the knowledge base setup
try:
    print("📚 Loading knowledge base module...")
    from knowledge_base_postgres import PostgreSQLKnowledgeBase
    
    print("🔌 Connecting to PostgreSQL...")
    kb = PostgreSQLKnowledgeBase(db_url=db_url, kb_folder="KB_")
    
    if not kb.connect_database():
        print("❌ Failed to connect to database")
        print("Please check:")
        print("  1. PostgreSQL is running")
        print("  2. Password is correct")
        print("  3. Database 'symptom_kb' exists")
        sys.exit(1)
    
    print("✅ Connected to database successfully!")
    print()
    
    print("📋 Initializing knowledge base...")
    if kb.initialize_knowledge_base():
        print()
        print("=" * 60)
        print("✅ KNOWLEDGE BASE INITIALIZATION COMPLETE!")
        print("=" * 60)
        print(f"📊 Loaded {len(kb.documents)} medical documents")
        print()
        print("Next steps:")
        print("  1. Install Ollama: https://ollama.ai/download")
        print("  2. Download model: ollama pull mistral:7b-instruct")
        print("  3. Start API server: py simple_api.py")
        print()
    else:
        print("❌ Failed to initialize knowledge base")
        print("Check the error messages above for details")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ Import error: {e}")
    print()
    print("Make sure all dependencies are installed:")
    print("  pip install psycopg2-binary sqlalchemy sentence-transformers")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
