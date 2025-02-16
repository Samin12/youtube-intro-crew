from typing import List
from crewai import Agent
from ..utils.llm_utils import LLMProvider
from ..rag_service import RAGService

class CredibilityInjector:
    """Agent responsible for injecting credibility markers into content."""
    
    def __init__(self, rag_service: RAGService):
        self.rag = rag_service
        self.llm = LLMProvider.get_chat_model(temperature=0.7)
    
    def inject_credibility(self, content: str) -> str:
        """
        Inject credibility markers into the content based on proven patterns.
        
        Key Credibility Elements:
        1. Duration of Expertise: How long you've been an expert
        2. People Helped: Number of people you've helped with this specific problem
        3. Personal Results: Big results you've achieved yourself
        4. Client Results: Notable results achieved by clients
        5. Research Depth: Emphasize thorough research and testing
        
        Example:
        "So in this video im going to show you how I used this strategy to generate 10,000 new followers in a day"
        "And I know this works because my clients have gone from complete beginners to performing on stage in 90 days using this:"
        """
        # Get relevant credibility patterns from our knowledge base
        context = self.rag.query(
            "credibility examples client results personal achievements research depth expertise duration",
            num_results=5
        )
        
        prompt = f'''Enhance this content by naturally weaving in credibility markers.
        
        Required Credibility Elements (include at least 2):
        1. Mention how long you've been an expert in your field
        2. Mention how many people you've already helped solve this specific problem
        3. Share a big personal result that relates to the topic
        4. Share impressive client results that relate to the topic
        5. Emphasize the depth of research/testing done
        
        Rules:
        1. NEVER make up fake credentials or results
        2. Weave credibility naturally into the content (not as standalone statements)
        3. Use specific numbers when possible (e.g., "helped 100+ creators" vs "helped many creators")
        4. Focus on results that directly relate to the video topic
        5. Make the research/testing sound comprehensive
        
        Reference Examples:
        {context}
        
        Original Content:
        {content}
        
        Enhanced Content with Credibility (maintain the same basic structure):'''
        
        response = self.llm.predict(prompt)
        return response
    
    def analyze_credibility(self, content: str) -> List[dict]:
        """
        Analyze content for credibility markers and suggest improvements.
        Focuses on expertise, client results, personal results, and research depth.
        """
        # Get credibility best practices from our knowledge base
        context = self.rag.query(
            "credibility expertise duration client results personal achievements research depth examples",
            num_results=3
        )
        
        prompt = f'''Analyze this content for credibility and trust-building elements.
        
        Required Elements to Check:
        1. Expertise Duration: Mentions of how long they've been an expert
        2. People Helped: References to number of people helped
        3. Personal Results: Specific achievements related to topic
        4. Client Results: Notable client successes related to topic
        5. Research Depth: Evidence of thorough research/testing
        
        Reference Examples:
        {context}
        
        Content to Analyze:
        {content}
        
        Provide analysis in this JSON format:
        {{
            "existing_markers": [
                {{
                    "type": "expertise_duration|people_helped|personal_result|client_result|research_depth",
                    "text": "exact text from content",
                    "strength": "strong|medium|weak",
                    "impact": "how this builds credibility"
                }}
            ],
            "missing_elements": [
                {{
                    "type": "expertise_duration|people_helped|personal_result|client_result|research_depth",
                    "suggestion": "specific suggestion with example",
                    "placement": "where to naturally weave it in"
                }}
            ],
            "improvement_score": 0-100,
            "priority_fixes": ["list of top 3 most important fixes"],
            "credibility_checklist": {{
                "expertise_mentioned": true|false,
                "client_results_included": true|false,
                "personal_results_shared": true|false,
                "research_emphasized": true|false,
                "naturally_woven": true|false
            }}
        }}'''
        
        response = self.llm.predict(prompt)
        return response
