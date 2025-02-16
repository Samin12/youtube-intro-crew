from crewai import Agent
from langchain.prompts import PromptTemplate
import os

class ApplicationAgent(Agent):
    """Agent responsible for creating practical application sections."""
    
    def __init__(self, llm, rag_service):
        """Initialize the application agent.
        
        Args:
            llm: Language model to use
            rag_service: RAG service for retrieving relevant content
        """
        prompt_path = os.path.join("prompts", "application.txt")
        with open(prompt_path, "r") as f:
            application_template = f.read()
            
        application_prompt = PromptTemplate(
            template=application_template,
            input_variables=["point", "context", "example"]
        )
        
        super().__init__(
            name="Application Writer",
            llm=llm,
            system_message=application_prompt.format(
                point="",  # Will be provided during task execution
                context="",
                example=""
            ),
            verbose=True
        )
        self.rag_service = rag_service
        
    def get_application_examples(self, context):
        """Get relevant application examples from the RAG service."""
        query = f"Find examples of practical applications related to: {context}"
        return self.rag_service.query(query)
