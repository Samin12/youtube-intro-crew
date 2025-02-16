from crewai import Crew, Task
from typing import List, Dict, Any
import os
from datetime import datetime

from src.utils.llm_utils import LLMProvider
from src.rag_service import RAGService
import config.config as config
from src.agents.middle_point.example_brick import ExampleBrickAgent, ExampleType
from src.agents.middle_point.transition import TransitionAgent
from src.agents.middle_point.application import ApplicationAgent
from src.agents.middle_point.base_reviewer import BaseReviewer

class MiddlePointCrew:
    """Crew responsible for generating middle points of a video script."""
    
    def __init__(self):
        """Initialize the middle point crew."""
        self.llm = LLMProvider.get_chat_model()
        self.rag = RAGService(config.PINECONE_INDEX_NAME)
        
        # Initialize all agents
        self.story_agent = ExampleBrickAgent(self.llm, self.rag, ExampleType.STORY)
        self.metaphor_agent = ExampleBrickAgent(self.llm, self.rag, ExampleType.METAPHOR)
        self.framework_agent = ExampleBrickAgent(self.llm, self.rag, ExampleType.FRAMEWORK)
        self.transition_agent = TransitionAgent(self.llm, self.rag)
        self.application_agent = ApplicationAgent(self.llm, self.rag)
        
        # Initialize reviewers
        self.example_reviewer = BaseReviewer(self.llm, self.rag, "example")
        self.transition_reviewer = BaseReviewer(self.llm, self.rag, "transition")
        self.application_reviewer = BaseReviewer(self.llm, self.rag, "application")
        
    def generate_middle_points(self, video_points: List[Dict[str, Any]], example_type: ExampleType) -> str:
        """Generate middle points for a video script.
        
        Args:
            video_points: List of dictionaries containing point details
            example_type: Type of example to use (story, metaphor, or framework)
            
        Returns:
            Generated middle points content
        """
        tasks = []
        total_points = len(video_points)
        
        for i, point in enumerate(video_points):
            # Select the appropriate example agent
            if example_type == ExampleType.STORY:
                example_agent = self.story_agent
            elif example_type == ExampleType.METAPHOR:
                example_agent = self.metaphor_agent
            else:
                example_agent = self.framework_agent
                
            # Create example brick task
            example_task = Task(
                description=f"Create an example for point {i+1}: {point['title']}",
                agent=example_agent,
                context=point
            )
            tasks.append(example_task)
            
            # Create example review task
            example_review_task = Task(
                description=f"Review the example for point {i+1}",
                agent=self.example_reviewer,
                context={"content": example_task.output, "point": point}
            )
            tasks.append(example_review_task)
            
            # Create transition task if not the last point
            if i < total_points - 1:
                transition_task = Task(
                    description=f"Create transition from point {i+1} to point {i+2}",
                    agent=self.transition_agent,
                    context={
                        "current_point": point,
                        "next_point": video_points[i+1]
                    }
                )
                tasks.append(transition_task)
                
                # Create transition review task
                transition_review_task = Task(
                    description=f"Review the transition from point {i+1} to {i+2}",
                    agent=self.transition_reviewer,
                    context={"content": transition_task.output}
                )
                tasks.append(transition_review_task)
            
            # Create application task
            application_task = Task(
                description=f"Create practical application for point {i+1}",
                agent=self.application_agent,
                context={
                    "point": point,
                    "example": example_task.output
                }
            )
            tasks.append(application_task)
            
            # Create application review task
            application_review_task = Task(
                description=f"Review the application for point {i+1}",
                agent=self.application_reviewer,
                context={"content": application_task.output}
            )
            tasks.append(application_review_task)
        
        # Create and run the crew
        crew = Crew(
            agents=[
                self.story_agent,
                self.metaphor_agent,
                self.framework_agent,
                self.transition_agent,
                self.application_agent,
                self.example_reviewer,
                self.transition_reviewer,
                self.application_reviewer
            ],
            tasks=tasks,
            verbose=True
        )
        
        result = crew.kickoff()
        
        # Save the output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join("outputs", "middle_points", f"generated_middle_points_{timestamp}.md")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(result)
            
        return result
