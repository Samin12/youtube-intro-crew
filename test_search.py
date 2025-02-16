import os
from dotenv import load_dotenv
from pinecone import Pinecone
from src.utils.llm_utils import LLMProvider

load_dotenv()

# Initialize
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("youtube-intros-bert")
model = LLMProvider.get_embeddings_model()

# Test queries
queries = [
    "Tell me about the BENS framework for YouTube success",
    "What are some effective hooks and examples?",
    "How to handle refunds and customer support?",
    "Tell me about AI implementation and CRM features",
    "What are the best practices for YouTube thumbnails?"
]

print("Testing similarity search...")
for query in queries:
    # Generate query embedding
    query_embedding = model.embed_query(query)
    
    # Search in Pinecone
    results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
    
    print(f"\nQuery: {query}")
    print("Top matches:")
    for match in results["matches"]:
        print(f"Score: {match['score']:.4f}")
        if 'metadata' in match and match['metadata']:
            metadata = match['metadata']
            if 'type' in metadata:
                print(f"Type: {metadata['type']}")
            if 'source' in metadata:
                print(f"Source: {metadata['source']}")
            if 'text' in metadata:
                print(f"Preview: {metadata['text'][:200]}...")
        print("---")
