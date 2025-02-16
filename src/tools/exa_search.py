import os
from typing import List, Dict, Any, Tuple
from openai import OpenAI
import json

class ExaSearch:
    """Tool for searching web content using Exa API."""
    
    def __init__(self):
        """Initialize Exa client."""
        api_key = os.getenv('EXAAI_API_KEY')
        if not api_key:
            raise ValueError("EXAAI_API_KEY environment variable not set")
        self.client = OpenAI(
            base_url="https://api.exa.ai",
            api_key=api_key
        )
        
    def get_answer(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get an answer from Exa using chat completions.
        
        Args:
            query: Question to ask
            
        Returns:
            Tuple containing:
                - str: Generated answer
                - List[Dict]: List of citations with metadata
        """
        try:
            completion = self.client.chat.completions.create(
                model="exa",
                messages=[
                    {"role": "user", "content": query}
                ]
            )
            # Parse the response content as JSON
            try:
                response_data = json.loads(completion.choices[0].message.content)
            except json.JSONDecodeError:
                # If response is not JSON, treat the entire content as the answer
                return completion.choices[0].message.content, []
            
            # Extract answer and citations
            answer = response_data.get('answer', '')
            citations = response_data.get('citations', [])
            
            # Clean up citations to only include relevant fields
            cleaned_citations = [{
                'url': citation.get('url'),
                'title': citation.get('title'),
                'author': citation.get('author'),
                'published_date': citation.get('publishedDate'),
                'score': citation.get('score')
            } for citation in citations]
            
            return answer, cleaned_citations
        except Exception as e:
            print(f"Error getting answer from Exa: {str(e)}")
            return "", []
