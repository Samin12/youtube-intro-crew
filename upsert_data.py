import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
from src.utils.llm_utils import LLMProvider
from pathlib import Path

load_dotenv()

def split_content(content):
    """Split content into meaningful chunks using various delimiters."""
    # Split by multiple possible section markers
    delimiters = [
        '\n\n', # Double newline
        '\n#', # Markdown headers
        '\n1.', '\n2.', '\n3.', '\n4.', '\n5.', # Numbered lists
        'Example:', 'Why:', 'How:', # Common section markers
        'Step 1:', 'Step 2:', 'Step 3:' # Step markers
    ]
    
    chunks = [content]
    for delimiter in delimiters:
        new_chunks = []
        for chunk in chunks:
            if delimiter in chunk:
                new_chunks.extend([c.strip() for c in chunk.split(delimiter) if c.strip()])
            else:
                new_chunks.append(chunk)
        chunks = new_chunks
    
    # Filter out very short chunks and merge very small ones
    MIN_CHUNK_LENGTH = 100
    filtered_chunks = []
    current_chunk = ""
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < MIN_CHUNK_LENGTH:
            current_chunk += " " + chunk if current_chunk else chunk
        else:
            if current_chunk:
                filtered_chunks.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        filtered_chunks.append(current_chunk)
    
    return filtered_chunks

def read_file(file_path, file_type):
    """Read a file and return its content with metadata."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            print(f"Successfully read {file_type} from {file_path}")
            
            # For dumps.txt, split into meaningful chunks
            if "dumps.txt" in str(file_path):
                chunks = split_content(content)
                print(f"Split into {len(chunks)} chunks")
                return [{
                    "content": chunk,
                    "source": f"{file_path}#chunk{i}",
                    "type": f"{file_type}_chunk{i}"
                } for i, chunk in enumerate(chunks)]
            
            return [{
                "content": content,
                "source": str(file_path),
                "type": file_type
            }]
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return []

def read_json_file(file_path, file_type):
    """Read a JSON file and return its content with metadata."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            # Convert JSON to string representation
            content = json.dumps(data, indent=2)
            print(f"Successfully read {file_type} JSON from {file_path}")
            return [{
                "content": content,
                "source": str(file_path),
                "type": file_type
            }]
    except Exception as e:
        print(f"Error reading JSON {file_path}: {e}")
        return []

def collect_all_files(project_root):
    """Collect all relevant files from the project."""
    all_data = []
    
    # Project structure
    file_mappings = {
        # Personal data
        (project_root.parent / "personaldump.txt"): "personal_info",
        
        # YouTube Intro Crew files
        (project_root / "outputs").glob("*.md"): "generated_intro",
        (project_root / "prompts").glob("*.txt"): "prompt",
        (project_root / "config/testprompts.txt"): "test_prompt",
        (project_root / "anecdote.txt"): "content",
        (project_root / "annotation.txt"): "content",
        (project_root / "hook.txt"): "content",
        (project_root / "intro.txt"): "content",
        (project_root / "review.txt"): "content",
        (project_root / "setup.txt"): "content",
        (project_root / "transition.txt"): "content",
        
        # GPT4O Finetuning files
        (project_root.parent / "gpt4o-finetuning/content_dumps/dumps.txt"): "training_data",
        (project_root.parent / "gpt4o-finetuning/dumps.txt"): "training_data",
        (project_root.parent / "gpt4o-finetuning/generated_intros.json"): "generated_data",
        (project_root.parent / "gpt4o-finetuning/processed_data/concepts.json"): "concept_data",
        (project_root.parent / "gpt4o-finetuning/processed_data/examples.json"): "example_data",
        (project_root.parent / "gpt4o-finetuning/processed_data/guidelines.json"): "guideline_data",
        (project_root.parent / "gpt4o-finetuning/processed_data/templates.json"): "template_data",
    }
    
    # Process each file/pattern
    for file_pattern, file_type in file_mappings.items():
        if isinstance(file_pattern, Path):
            if file_pattern.suffix == '.json':
                all_data.extend(read_json_file(file_pattern, file_type))
            else:
                all_data.extend(read_file(file_pattern, file_type))
        else:
            # Handle glob patterns
            for file_path in file_pattern:
                if file_path.suffix == '.json':
                    all_data.extend(read_json_file(file_path, file_type))
                else:
                    all_data.extend(read_file(file_path, file_type))
    
    return all_data

def main():
    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("youtube-intros-bert")
    
    # Initialize embedding model
    model = LLMProvider.get_embeddings_model()
    
    # Read prompts
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Collect all files
    all_data = collect_all_files(project_root)
    print(f"Found {len(all_data)} total files")
    
    # Generate embeddings and upsert in batches
    batch_size = 50
    for i in range(0, len(all_data), batch_size):
        batch = all_data[i:i + batch_size]
        
        # Generate embeddings for the batch
        texts = [item["content"] for item in batch]
        embeddings = model.embed_documents(texts)
        
        # Prepare vectors for upserting
        vectors = []
        for j, (embedding, item) in enumerate(zip(embeddings, batch)):
            vector_id = f"{item['type']}_{i+j}"
            vectors.append((vector_id, embedding, {
                "text": item["content"][:1000],  # Limit metadata size
                "source": item["source"],
                "type": item["type"]
            }))
        
        # Upsert to Pinecone
        index.upsert(vectors=vectors)
        print(f"Upserted batch {i//batch_size + 1}, vectors {i+1} to {min(i+batch_size, len(all_data))}")

if __name__ == "__main__":
    main()
