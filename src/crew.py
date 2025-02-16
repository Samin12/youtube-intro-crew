import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from .utils.llm_utils import LLMProvider
from .rag_service import RAGService
from .agents.credibility_injector import CredibilityInjector
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.config import config, ConfigurationError

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Load environment variables
load_dotenv()

class YouTubeIntroCrew:
    def __init__(self):
        """Initialize the YouTube Intro Crew with all necessary agents."""
        try:
            # Initialize LLM with fallback capabilities
            self.llm = LLMProvider.get_chat_model()
            
            # Initialize RAG service
            self.rag = RAGService(config.PINECONE_INDEX_NAME)
            
            # Initialize Exa search tool
            from .tools.exa_search import ExaSearch
            self.exa = ExaSearch()
            
            self._create_agents()
            logger.info("Successfully initialized YouTube Intro Crew")
        except Exception as e:
            logger.error(f"Failed to initialize crew: {str(e)}")
            raise
    
    def _load_prompt(self, filename):
        """Load a prompt file from the prompts directory."""
        prompt_path = Path(config.PROMPTS_DIR) / filename
        try:
            with open(prompt_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load prompt {filename}: {str(e)}")
            raise

    def _create_agents(self):
        """Create all the specialized agents for the crew."""
        try:
            base_instructions = """
                You are an expert at creating engaging YouTube video intros. Follow these rules:
                1. NEVER return placeholder text
                2. ALWAYS generate real, high-quality content
                3. ALWAYS format output exactly as requested
                4. FOCUS on the video outline and task
                5. BE creative and engaging
                6. ALWAYS include credibility markers naturally
                7. USE natural, conversational language
                8. MAKE viewers curious to learn more
                
                When asked to generate content:
                1. READ the task description carefully
                2. IDENTIFY the required output format
                3. GENERATE content that matches the format exactly
                4. REVIEW your output to ensure it meets all requirements
                5. RETURN only the final, formatted content
                """
            
            # Load all prompts first
            hook_prompt = self._load_prompt('hook_prompt.txt')
            anecdote_prompt = self._load_prompt('anecdote_prompt.txt')
            setup_prompt = self._load_prompt('setup_prompt.txt')
            intro_prompt = self._load_prompt('intro_prompt.txt')
            transition_prompt = self._load_prompt('transition_prompt.txt')
            reviewer_prompt = self._load_prompt('reviewer_prompt.txt')
            
            # Create hook writer agent
            self.hook_writer = Agent(
                role='Hook Writer',
                goal='Write compelling one-line hooks that grab attention',
                backstory=hook_prompt,
                system_prompt=hook_prompt,
                llm=self.llm,
                verbose=True
            )
            logger.debug("Created Hook Writer agent")
            
            # Create anecdote writer agent
            self.anecdote_writer = Agent(
                role='Anecdote Writer',
                goal='Write credibility-building anecdotes and stories',
                backstory=anecdote_prompt,
                system_prompt=anecdote_prompt,
                llm=self.llm,
                verbose=True
            )
            logger.debug("Created Anecdote Writer agent")
            
            # Create setup writer agent
            self.setup_writer = Agent(
                role='Setup Writer',
                goal='Write clear problem/result statements',
                backstory=setup_prompt,
                system_prompt=setup_prompt,
                llm=self.llm,
                verbose=True
            )
            logger.debug("Created Setup Writer agent")
            
            # Create intro writer agent
            self.intro_writer = Agent(
                role='Intro Writer',
                goal='Write complete, engaging video introductions',
                backstory=intro_prompt,
                system_prompt=intro_prompt,
                llm=self.llm,
                verbose=True
            )
            logger.debug("Created Intro Writer agent")
            
            # Create transition writer agent
            self.transition_writer = Agent(
                role='Transition Writer',
                goal='Write smooth transitions into main content',
                backstory=transition_prompt,
                system_prompt=transition_prompt,
                llm=self.llm,
                verbose=True
            )
            logger.debug("Created Transition Writer agent")
            
            # Create content reviewer agent
            self.content_reviewer = Agent(
                role='Content Reviewer',
                goal='Review and polish complete intros',
                backstory=reviewer_prompt,
                system_prompt=reviewer_prompt,
                llm=self.llm,
                verbose=True
            )
            logger.debug("Created Content Reviewer agent")
            logger.debug("Created Editor agent")
            
        except Exception as e:
            logger.error(f"Failed to create agents: {str(e)}")
            raise
    
    def _create_tasks(self, video_outline):
        """Create tasks for the crew based on the video outline."""
        tasks = []
        
        # Get relevant examples and style guidelines for hook
        hook_examples = self.rag.enhance_prompt_with_examples("hook", video_outline)
        style_guidelines = self.rag.get_style_guidelines(video_outline)
        
        # Get relevant insights using Exa
        try:
            answer, citations = self.exa.get_answer(
                f"What makes a YouTube video intro engaging and successful? Focus on {video_outline}"
            )
            
            exa_examples = "\nInsights from research:\n" + answer
            
            if citations:
                exa_examples += "\n\nSources:\n"
                for citation in citations:
                    if citation.get('title') and citation.get('url'):
                        exa_examples += f"- {citation['title']}: {citation['url']}\n"
        except Exception as e:
            logger.warning(f"Failed to get Exa insights: {str(e)}")
            exa_examples = ""
        
        # Hook Task
        hook_task = Task(
            description=f"""
                Create a compelling hook for a YouTube video with this outline:
                {video_outline}
                
                Follow the exact instructions in your backstory to create a hook.
                Choose the most appropriate hook type and make it one line that
                makes people curious to know more.
                
                {hook_examples}
                
                {style_guidelines}
                
                Here are some additional examples from successful YouTube videos:
                {exa_examples}
                
                Output format:
                [HOOK] Your hook text here
            """,
            agent=self.hook_writer,
            expected_output="A single-line hook that grabs attention and makes viewers curious",
            output_file="hook.txt"
        )
        tasks.append(hook_task)
        
        # Get relevant examples and style guidelines for anecdote
        anecdote_examples = self.rag.enhance_prompt_with_examples("anecdote", video_outline)
        anecdote_guidelines = self.rag.get_style_guidelines(f"anecdote {video_outline}")
        
        # Anecdote Task
        anecdote_task = Task(
            description=f"""
                Based on this video outline:
                {video_outline}
                
                Previous hook: {{hook_task.output}}
                
                Follow your backstory instructions to create a credibility-building
                element that can be woven naturally into the intro. Choose the most
                appropriate way to inject credibility from your options.
                
                {anecdote_examples}
                
                {anecdote_guidelines}
                
                Output format:
                [CREDIBILITY] Your credibility text here
            """,
            agent=self.anecdote_writer,
            expected_output="A credibility-building anecdote or story that enhances the intro",
            output_file="anecdote.txt"
        )
        tasks.append(anecdote_task)
        
        # Setup Task
        setup_task = Task(
            description=f"""
                Using this outline: {video_outline}
                
                Previous content:
                Hook: {{hook_task.output}}
                Credibility: {{anecdote_task.output}}
                
                Write the problem/result section following your backstory instructions exactly.
                Make sure it flows naturally with the previous content.
                
                Output format:
                [SETUP] Your setup text here
            """,
            agent=self.setup_writer,
            expected_output="A clear problem/result statement that resonates with viewers",
            output_file="setup.txt"
        )
        tasks.append(setup_task)
        
        # Intro Task
        intro_task = Task(
            description=f"""
                Video outline: {video_outline}
                
                Previous content:
                Hook: {{hook_task.output}}
                Credibility: {{anecdote_task.output}}
                Setup: {{setup_task.output}}
                
                Create a complete intro that follows the exact structure and guidelines
                in your backstory. Incorporate all previous elements naturally.
                
                Output format:
                [INTRO]
                Your complete intro text here
            """,
            agent=self.intro_writer,
            expected_output="A well-structured intro that incorporates all elements effectively",
            output_file="intro.txt"
        )
        tasks.append(intro_task)
        
        # Transition Task
        transition_task = Task(
            description=f"""
                Complete intro so far:
                {{intro_task.output}}
                
                Create a transition that leads into the main content.
                Follow your backstory instructions exactly and avoid forbidden phrases.
                
                Output format:
                [TRANSITION] Your transition text here
            """,
            agent=self.transition_writer,
            expected_output="A natural transition that maintains engagement",
            output_file="transition.txt"
        )
        tasks.append(transition_task)
        
        # Review Task
        review_task = Task(
            description=f"""
                Complete intro with transition:
                {{intro_task.output}}
                {{transition_task.output}}
                
                Review the complete intro using your checklist and criteria.
                Make any necessary improvements to ensure it meets all quality standards.
                
                Output format:
                [REVIEW]
                Your complete, polished intro here
            """,
            agent=self.content_reviewer,
            expected_output="A polished, engaging YouTube video intro",
            output_file="review.txt"
        )
        tasks.append(review_task)

        return tasks
    
    def generate_intro(self, video_outline: str, output_file: str = None) -> str:
        """
        Generate a YouTube video intro based on the provided outline.
        
        Args:
            video_outline: Brief outline of the video content
            output_file: Path to save the output. If None, a default name will be used.
            
        Returns:
            The complete, polished intro script
            
        Raises:
            ConfigurationError: If there are issues with generation
        """
        try:
            # Create default output file if none provided
            if output_file is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = config.OUTPUTS_DIR / f"generated_intro_{timestamp}.md"
            
            # Create tasks for the crew
            tasks = self._create_tasks(video_outline)
            
            # Create the crew with all specialized agents
            crew = Crew(
                agents=[
                    self.hook_writer,
                    self.anecdote_writer,
                    self.setup_writer,
                    self.intro_writer,
                    self.transition_writer,
                    self.content_reviewer
                ],
                tasks=tasks,
                verbose=True
            )
            
            # Execute the crew's tasks
            result = crew.kickoff()
            
            # Save the output
            with open(output_file, 'w') as f:
                f.write(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate intro: {str(e)}")
            raise ConfigurationError(f"Intro generation failed: {str(e)}")
            
            markdown_content = f"""# YouTube Video Intro Generation

## Video Outline
```
{video_outline}
```

## Generated Content

### Final Script
{final_intro}

### Relevant Examples Used
```
{examples_text}
```

### Style Guidelines Applied
```
{style_examples}
```

### Credibility Analysis
- Credibility markers included:
  - Years of experience
  - Number of people helped
  - Data and research backing
  - Case studies and proven results
  - Real-world success examples
"""
            
            output_path.write_text(markdown_content)
            logger.info(f"Saved output to: {output_path}")
            
            return final_intro
            
        except Exception as e:
            logger.error(f"Failed to generate intro: {str(e)}")
            raise ConfigurationError(f"Intro generation failed: {str(e)}")
    
    def test_rag_connection(self, query: str = "engaging youtube intro", k: int = 3) -> List[str]:
        """
        Test the RAG connection by retrieving examples from Pinecone.
        
        Args:
            query: Search query to test with
            k: Number of examples to retrieve
            
        Returns:
            List of retrieved examples
        """
        try:
            # Get relevant examples
            docs = self.rag.get_relevant_content(query, k=k)
            
            # Format results
            results = []
            for i, doc in enumerate(docs, 1):
                results.append(f"Example {i}:\n{doc.page_content}\n")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to test RAG connection: {str(e)}")
            raise ConfigurationError(f"RAG test failed: {str(e)}")
