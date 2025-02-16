import os
import logging
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

# Load environment variables from project root
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Custom exception for configuration-related errors."""
    pass

class Config:
    """Configuration manager for the YouTube Intro Crew project."""
    
    def __init__(self):
        # Project root directory
        self.ROOT_DIR = Path(__file__).parent.parent
        
        # Directory paths
        self.PROMPTS_DIR = self.ROOT_DIR / "prompts"
        self.OUTPUTS_DIR = self.ROOT_DIR / "outputs"
        self.SRC_DIR = self.ROOT_DIR / "src"
        
        # Model configuration
        self.MODEL_PROVIDER = "openai"  # Force OpenAI for fine-tuned model
        
        # Model names for different providers
        self.MODELS = {
            "openai": "ft:gpt-4o-2024-08-06:personal:youtube-intro-expert:B1Od8G5y",  # Our fine-tuned model
            "openai_backup": "gpt-4-turbo-preview",  # Backup model
            "anthropic": "anthropic/claude-3-sonnet-20240229",
            "openrouter": "anthropic/claude-3-sonnet-20240229"
        }
        
        # Pinecone configuration
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "youtube-intros-bert")
        
        # Override any environment variable
        os.environ["MODEL_PROVIDER"] = "openai"
        
        # Get the active model based on provider
        self.ACTIVE_MODEL = self.MODELS[self.MODEL_PROVIDER]
        
        # Initialize directories and prompts
        self._ensure_directories()
        self._initialize_prompt_files()
        
        # Validate environment
        self._validate_environment()
    
    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        try:
            for directory in [self.PROMPTS_DIR, self.OUTPUTS_DIR, self.SRC_DIR]:
                directory.mkdir(exist_ok=True)
                logger.info(f"Ensured directory exists: {directory}")
        except Exception as e:
            raise ConfigurationError(f"Failed to create directories: {str(e)}")
    
    def _initialize_prompt_files(self) -> None:
        """Initialize prompt file paths."""
        self.PROMPT_FILES = {
            "hook": self.PROMPTS_DIR / "hook_prompt.txt",
            "intro": self.PROMPTS_DIR / "intro_prompt.txt",
            "setup": self.PROMPTS_DIR / "setup_prompt.txt",
            "transition": self.PROMPTS_DIR / "transition_prompt.txt",
            "anecdote": self.PROMPTS_DIR / "anecdote_prompt.txt",
            "reviewer": self.PROMPTS_DIR / "reviewer_prompt.txt",
            "annotator": self.PROMPTS_DIR / "annotator.txt",
        }
        
        # Validate prompt files exist
        missing_files = [
            str(path) for path in self.PROMPT_FILES.values() 
            if not path.exists()
        ]
        if missing_files:
            raise ConfigurationError(
                f"Missing prompt files: {', '.join(missing_files)}"
            )
    
    def _validate_environment(self) -> None:
        """Validate required environment variables."""
        required_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ConfigurationError(
                f"Missing environment variables: {', '.join(missing_vars)}"
            )
    
    def get_prompt_content(self, prompt_name: str) -> str:
        """
        Get content of a prompt file by its name.
        
        Args:
            prompt_name: Name of the prompt file to read
            
        Returns:
            Content of the prompt file
            
        Raises:
            ConfigurationError: If prompt file doesn't exist or can't be read
        """
        try:
            file_path = self.PROMPT_FILES.get(prompt_name)
            if not file_path:
                raise ConfigurationError(f"Unknown prompt name: {prompt_name}")
            if not file_path.exists():
                raise ConfigurationError(f"Prompt file not found: {file_path}")
            
            content = file_path.read_text()
            logger.debug(f"Successfully read prompt file: {prompt_name}")
            return content
            
        except Exception as e:
            raise ConfigurationError(
                f"Failed to read prompt '{prompt_name}': {str(e)}"
            )
    
    def get_output_path(self, filename: str) -> Path:
        """
        Get full path for an output file.
        
        Args:
            filename: Name of the output file
            
        Returns:
            Full path in the outputs directory
        """
        return self.OUTPUTS_DIR / filename

# Create global config instance
config = Config()
