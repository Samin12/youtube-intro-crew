import os
from dotenv import load_dotenv
from pinecone import Pinecone
from src.utils.llm_utils import LLMProvider

load_dotenv()

# Initialize embedding model
model = LLMProvider.get_embeddings_model()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("youtube-intros-bert")

# Test texts
test_texts = [
    "Create an engaging YouTube intro for a tech tutorial",
    "Make a dramatic intro for a gaming channel",
    "Design a professional intro for a business vlog"
]

# Generate and store embeddings
for i, text in enumerate(test_texts):
    # Generate embedding
    embedding = model.embed_query(text)
    print(f"Text {i+1} embedding dimension: {len(embedding)}")
    
    # Store in Pinecone
    index.upsert(vectors=[(f"test_{i}", embedding, {"text": text})])
    print(f"Stored embedding {i+1} in Pinecone")

# Test similarity search
query = "Create a YouTube intro for a programming tutorial"
query_embedding = model.embed_query(query)

# Search in Pinecone
results = index.query(vector=query_embedding, top_k=2, include_metadata=True)
print("\nSimilarity Search Results:")
for match in results["matches"]:
    print(f"Score: {match['score']:.4f} | Text: {match['metadata']['text']}")
