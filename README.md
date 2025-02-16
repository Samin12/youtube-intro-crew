# YouTube Intro Crew

An AI-powered system that generates engaging YouTube video introductions using specialized AI agents.

## Features

- **Specialized Agents**: Each aspect of the intro is handled by a dedicated AI agent
  - Hook Writer: Creates attention-grabbing opening lines
  - Intro Writer: Crafts the overall introduction
  - Setup Writer: Frames problems and desired results
  - Transition Writer: Creates smooth transitions
  - Anecdote Writer: Injects credibility through stories
  - Content Reviewer: Ensures quality and engagement

- **Structured Output**: Generates well-formatted Markdown files with both the input outline and final script
- **Configurable**: Easy to modify prompts and agent behaviors
- **Error Handling**: Robust error handling and logging

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables in `.env`:
   ```
   ANTHROPIC_API_KEY=your_key_here
   ```

## Usage

```python
from src.crew import YouTubeIntroCrew

# Create the crew
crew = YouTubeIntroCrew()

# Define your video outline
outline = """
Title: "Your Video Title"
Key Points:
1. Point One
2. Point Two
3. Point Three
Target Audience: Your target audience
"""

# Generate the intro
result = crew.generate_intro(outline)
```

The generated intro will be saved in `outputs/generated_intro.md`.

## Project Structure

```
youtube_intro_crew/
├── config/
│   └── config.py         # Configuration and file paths
├── prompts/
│   ├── anecdote_prompt.txt
│   ├── hook_prompt.txt
│   ├── intro_prompt.txt
│   ├── reviewer_prompt.txt
│   ├── setup_prompt.txt
│   └── transition_prompt.txt
├── src/
│   ├── crew.py          # Main crew implementation
│   └── main.py          # Runner script
└── outputs/
    └── generated_intro.md
```

## Requirements

- Python 3.8+
- CrewAI
- Anthropic Claude API access
- Required Python packages (see requirements.txt)
