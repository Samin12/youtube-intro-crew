import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create a new index with dimension 1536 for OpenAI ada-002
index_name = "youtube-intros-bert"

# Always recreate the index
try:
    pc.delete_index(index_name)
    print(f"Deleted existing index: {index_name}")
except Exception as e:
    print(f"Error deleting index: {e}")

# Create new index
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print(f"Created new index: {index_name}")
else:
    print(f"Index {index_name} already exists")
