from crewai import Agent
from langchain.prompts import PromptTemplate
import os

class BaseReviewer(Agent):
    """Base reviewer agent that can be extended for specific components."""
    
    def __init__(self, llm, rag_service, component_name):
        """Initialize the base reviewer.
        
        Args:
            llm: Language model to use
            rag_service: RAG service for retrieving relevant content
            component_name: Name of the component being reviewed (e.g., "story", "metaphor")
        """
        prompt_path = os.path.join("prompts", "reviewer_prompt.txt")
        with open(prompt_path, "r") as f:
            review_template = f.read()
            
        review_prompt = PromptTemplate(
            template=review_template,
            input_variables=["content", "component_name"]
        )
        
        super().__init__(
            name=f"{component_name.title()} Reviewer",
            llm=llm,
            system_message=review_prompt.format(
                content="",  # Will be provided during task execution
                component_name=component_name
            ),
            verbose=True
        )
        self.rag_service = rag_service
        self.component_name = component_name
