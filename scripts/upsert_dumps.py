import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import from src
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.rag_service import RAGService
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Upsert dumps content to Pinecone."""
    try:
        # Initialize RAG service
        rag = RAGService(config.PINECONE_INDEX_NAME)
        
        # Read dumps file
        dumps_path = Path(parent_dir).parent / "gpt4o-finetuning" / "content_dumps" / "dumps.txt"
        with open(dumps_path, "r") as f:
            content = f.read()
        
        # Split content into chunks
        chunks = rag.text_splitter.split_text(content)
        
        # Add each chunk to Pinecone
        for i, chunk in enumerate(chunks):
            rag.add_example(
                content=chunk,
                metadata={
                    "source": "dumps.txt",
                    "chunk_id": i,
                    "content_type": "youtube_intro_guide",
                    "task_type": "intro_generation"
                }
            )
            logger.info(f"Upserted chunk {i+1}/{len(chunks)}")
        
        logger.info(f"Successfully upserted {len(chunks)} chunks to Pinecone")
        
    except Exception as e:
        logger.error(f"Failed to upsert dumps: {str(e)}")
        raise

if __name__ == "__main__":
    main()
