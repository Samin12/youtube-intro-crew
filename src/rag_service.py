import os
import logging
from typing import List, Dict, Any
from datetime import datetime
from langchain.vectorstores import Pinecone
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.pinecone import Pinecone as PineconeVectorStore
from pinecone import Pinecone as PineconeClient
from .utils.llm_utils import LLMProvider

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self, index_name: str):
        """Initialize the RAG service with Pinecone and LangChain."""
        # Initialize Pinecone client
        self.pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = index_name
        
        # Initialize embeddings model with fallback
        self.embeddings = LLMProvider.get_embeddings_model()
        
        # Initialize LangChain's Pinecone vectorstore with personal namespace
        self.vector_store = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=self.embeddings,
            text_key="text",  # The key that contains the text in your Pinecone documents
            namespace="personal"  # Use the personal namespace
        )
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )

    def get_relevant_content(self, query: str, k: int = 3, **kwargs) -> List[Document]:
        """
        Retrieve relevant content from Pinecone based on the query.
        
        Args:
            query: The search query
            k: Number of results to return
            **kwargs: Additional arguments to pass to similarity_search
            
        Returns:
            List of relevant documents
        """
        try:
            # Try with OpenAI embeddings
            self.embeddings = LLMProvider.get_embeddings_model()
            self.vector_store._embedding = self.embeddings
            
            # Use LangChain's similarity search with customizable parameters
            results = self.vector_store.similarity_search(
                query,
                k=k,
                **kwargs
            )
            return results
            
        except Exception as e:
            # Try with OpenRouter embeddings
            logger.warning(f"Primary embeddings failed: {str(e)}. Trying fallback...")
            self.embeddings = LLMProvider.get_embeddings_model(use_fallback=True)
            self.vector_store._embedding = self.embeddings
            
            # Retry with OpenRouter
            results = self.vector_store.similarity_search(
                query,
                k=k,
                **kwargs
            )
            return results

    def enhance_prompt_with_examples(self, task_type: str, video_outline: str, k: int = 3) -> str:
        """
        Enhance a prompt with relevant examples based on the task type and video outline.
        
        Args:
            task_type: Type of content to generate (hook, intro, etc.)
            video_outline: The video outline to base the search on
            k: Number of examples to retrieve
            
        Returns:
            Enhanced prompt with relevant examples
        """
        # Create metadata filter for the task type
        filter = {"task_type": task_type}
        
        # Get relevant examples using similarity search with metadata filtering
        relevant_docs = self.get_relevant_content(
            query=video_outline,
            k=k,
            filter=filter
        )
        
        # Format examples for the prompt
        examples = "\n\n".join([
            f"Example {i+1}:\n{doc.page_content}"
            for i, doc in enumerate(relevant_docs)
        ])
        
        enhanced_prompt = f"""
        Here are some relevant {task_type} examples to inspire your work:
        
        {examples}
        
        Use these examples as inspiration while maintaining originality and relevance to the current task.
        Focus on incorporating successful elements while adapting them to your unique context.
        """
        
        return enhanced_prompt

    def get_style_guidelines(self, video_outline: str, k: int = 5) -> str:
        """
        Get style guidelines based on similar content.
        
        Args:
            video_outline: The video outline to base the search on
            k: Number of examples to retrieve
            
        Returns:
            Style guidelines derived from similar content
        """
        # Get relevant style guidelines using similarity search with metadata filtering
        relevant_docs = self.get_relevant_content(
            query=f"style guidelines for {video_outline}",
            k=k,
            filter={"content_type": "style_guidelines"}
        )
        
        # Extract and combine style insights
        guidelines = []
        for doc in relevant_docs:
            content = doc.page_content.strip()
            if content:
                guidelines.append(f"- {content}")
        
        return "Style Guidelines Based on Similar Content:\n\n" + "\n".join(guidelines)

    def add_example(self, content: str, metadata: Dict[str, Any]) -> None:
        """
        Add a new example to the vectorstore.
        
        Args:
            content: The content to add
            metadata: Metadata about the content (e.g., task_type, content_type, etc.)
        """
        # Split content into chunks if it's long
        chunks = self.text_splitter.split_text(content)
        
        # Add chunks to vectorstore
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=[metadata for _ in chunks]
        )

    def store_successful_intro(self, intro_content: str, video_outline: str, performance_metrics: Dict[str, Any] = None) -> None:
        """
        Store a successful intro in the vectorstore for future reference.
        
        Args:
            intro_content: The full intro content
            video_outline: The video outline it was based on
            performance_metrics: Optional metrics about the intro's performance
        """
        # Store the full intro
        self.add_example(
            content=intro_content,
            metadata={
                "content_type": "full_intro",
                "video_outline": video_outline,
                "performance_metrics": performance_metrics or {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Extract and store the hook separately
        if "[HOOK]" in intro_content:
            hook = intro_content.split("[HOOK]")[1].split("\n")[0].strip()
            self.add_example(
                content=hook,
                metadata={
                    "content_type": "hook",
                    "task_type": "hook",
                    "video_outline": video_outline,
                    "performance_metrics": performance_metrics or {},
                    "timestamp": datetime.now().isoformat()
                }
            )
