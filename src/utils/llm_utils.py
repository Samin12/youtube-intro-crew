import os
import logging
from typing import Optional, Any
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from openai import RateLimitError, APIError

logger = logging.getLogger(__name__)

class LLMProvider:
    """Utility class to handle LLM provider fallbacks."""
    
    @staticmethod
    def get_chat_model(model_name: str = "anthropic/claude-3-sonnet", temperature: float = 0.7, use_fallback: bool = False) -> ChatOpenAI:
        """
        Get a chat model with fallback to OpenRouter if OpenAI fails.
        
        Args:
            model_name: Name of the model to use
            temperature: Temperature for generation
            use_fallback: If True, use OpenRouter directly without trying OpenAI
            
        Returns:
            ChatOpenAI model
        """
        return ChatOpenAI(
            model=model_name,
            openai_api_key=os.getenv("OPENROUTER_API_KEY"),
            openai_api_base="https://openrouter.ai/api/v1",
            temperature=temperature,
            default_headers={
                "HTTP-Referer": "https://github.com/codeium/windsurf",
                "X-Title": "YouTube Intro Crew"
            }
        )

    @staticmethod
    def get_embeddings_model(model_name: str = "SeanLee97/mxbai-embed-large-v1-nli-matryoshka", use_fallback: bool = False) -> HuggingFaceEmbeddings:
        """
        Get an embeddings model using sentence-transformers.
        
        Args:
            model_name: Name of the sentence-transformers model
            use_fallback: Not used
            
        Returns:
            HuggingFaceEmbeddings model
        """
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'}
        )

    @staticmethod
    def with_fallback(func: callable, *args, **kwargs) -> Any:
        """
        Execute a function with OpenAI, falling back to OpenRouter if it fails.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function
        """
        try:
            # Try with OpenAI
            kwargs['api_key'] = os.getenv("OPENAI_API_KEY")
            return func(*args, **kwargs)
        except (RateLimitError, APIError) as e:
            logger.warning(f"OpenAI API error: {str(e)}. Falling back to OpenRouter.")
            # Try with OpenRouter
            kwargs['api_key'] = os.getenv("OPENROUTER_API_KEY")
            kwargs['api_base'] = "https://openrouter.ai/api/v1"
            kwargs['headers'] = {
                "HTTP-Referer": "https://github.com/codeium/windsurf",
                "X-Title": "YouTube Intro Crew"
            }
            return func(*args, **kwargs)
