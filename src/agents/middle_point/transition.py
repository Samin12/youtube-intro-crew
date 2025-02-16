from crewai import Agent
from langchain.prompts import PromptTemplate
import os

class TransitionAgent(Agent):
    """Agent responsible for creating smooth transitions between points."""
    
    def __init__(self, llm, rag_service):
        """Initialize the transition agent.
        
        Args:
            llm: Language model to use
            rag_service: RAG service for retrieving relevant content
        """
        prompt_path = os.path.join("prompts", "transition_prompt.txt")
        with open(prompt_path, "r") as f:
            transition_template = f.read()
            
        transition_prompt = PromptTemplate(
            template=transition_template,
            input_variables=["current_point", "next_point", "context"]
        )
        
        super().__init__(
            name="Transition Writer",
            llm=llm,
            system_message=transition_prompt.format(
                current_point="",  # Will be provided during task execution
                next_point="",
                context=""
            ),
            verbose=True
        )
        self.rag_service = rag_service
        
    def get_transition_examples(self, context):
        """Get relevant transition examples from the RAG service."""
        query = f"Find examples of smooth transitions in content about: {context}"
        return self.rag_service.query(query)
