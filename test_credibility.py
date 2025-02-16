#!/usr/bin/env python3

from src.crew import YouTubeIntroCrew
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def main():
    # Initialize the crew
    crew = YouTubeIntroCrew()

    # Test video outline
    outline = '''
    Title: How to Grow Your YouTube Channel Fast in 2025
    Topic: YouTube growth strategies that actually work
    Key Points:
    - Algorithm optimization techniques
    - Content strategy for rapid growth
    - Engagement tactics that multiply views
    - Monetization strategies for new creators
    '''

    # Generate intro
    try:
        result = crew.generate_intro(outline)
        print('\nGenerated Intro:\n', result)
    except Exception as e:
        logging.error(f"Error generating intro: {str(e)}")
        raise

if __name__ == "__main__":
    main()
