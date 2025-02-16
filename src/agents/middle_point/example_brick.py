from crewai import Agent
from langchain.prompts import PromptTemplate
import os
from enum import Enum

class ExampleType(Enum):
    STORY = "story"
    METAPHOR = "metaphor"
    FRAMEWORK = "framework"

class ExampleBrickAgent(Agent):
    """Agent responsible for creating example bricks using different types."""
    
    def __init__(self, llm, rag_service, example_type: ExampleType):
        """Initialize the example brick agent.
        
        Args:
            llm: Language model to use
            rag_service: RAG service for retrieving relevant content
            example_type: Type of example to generate (story, metaphor, or framework)
        """
        # Load the specific prompt for this example type
        prompt_path = os.path.join("prompts", f"{example_type.value}_brick.txt")
        with open(prompt_path, "r") as f:
            example_template = f.read()
            
        example_prompt = PromptTemplate(
            template=example_template,
            input_variables=["context", "point_number", "total_points"]
        )
        
        super().__init__(
            name=f"{example_type.value.title()} Creator",
            llm=llm,
            system_message=example_prompt.format(
                context="",  # Will be provided during task execution
                point_number=1,
                total_points=1
            ),
            verbose=True
        )
        self.rag_service = rag_service
        self.example_type = example_type
        
    def get_relevant_examples(self, context):
        """Get relevant examples from the RAG service."""
        query = f"Find examples of {self.example_type.value}s related to: {context}"
        return self.rag_service.query(query)
