import os
import sys
import logging
import argparse
from pathlib import Path

# Add parent directory to path to import from src
parent_dir = str(Path(__file__).parent.parent)
sys.path.append(parent_dir)

from src.crew import YouTubeIntroCrew
from config.config import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Generate YouTube video intros')
    parser.add_argument(
        '--model',
        choices=['openai', 'finetuned', 'finetuned2', 'claude'],
        default='openai',
        help='Model to use for generation: openai (GPT-4), finetuned/finetuned2 (custom models), or claude (via OpenRouter)'
    )
    return parser.parse_args()

def main():
    """Test generating a YouTube intro."""
    try:
        args = parse_args()
        
        # Set model based on argument
        if args.model == 'openai':
            os.environ["MODEL_PROVIDER"] = "openai"
            config.MODELS["openai"] = "gpt-4-turbo-preview"
        elif args.model == 'finetuned':
            os.environ["MODEL_PROVIDER"] = "openai"
            config.MODELS["openai"] = "ft:gpt-4o-2024-08-06:personal:youtube-intro-expert:B1Od8G5y"
        elif args.model == 'finetuned2':
            os.environ["MODEL_PROVIDER"] = "openai"
            config.MODELS["openai"] = "ft:gpt-4o-2024-08-06:ai-answer:samin-yt:B1QORJbq"
        elif args.model == 'claude':
            os.environ["MODEL_PROVIDER"] = "openrouter"
            config.MODELS["openrouter"] = "anthropic/claude-3-sonnet-20240229"
        
        # Initialize the crew
        crew = YouTubeIntroCrew()
        
        # Video outline for testing
        video_outline = """
        Title: "5 AI Tools That Will Make You $10,000/Month Working 2 Hours a Day"
        
        Key Points:
        - These are tools I personally use in my business
        - Each tool replaces 1-2 full-time employees
        - Combined they save over $20,000 in monthly salaries
        - I'll show exactly how to set them up
        - Real examples of how they work
        - ROI breakdown for each tool
        """
        
        # Generate the intro
        intro = crew.generate_intro(video_outline)
        logger.info(f"Generated intro using {args.model} model:\n{intro}")
        
    except Exception as e:
        logger.error(f"Failed to generate intro: {str(e)}")
        raise

if __name__ == "__main__":
    main()
