from crew import YouTubeIntroCrew

def main():
    # Create the crew
    crew = YouTubeIntroCrew()
    
    # Example video outline
    sample_outline = """
    Title: "The Complete Guide to AI in 2025: From Beginner to Business Owner"
    
    Key Sections:
    1. The New AI Landscape
       - Why 2025 is different
       - What actually matters now vs what doesn't
    
    2. The Only Foundation You Need
       - Basic prompt engineering
       - Learning to learn with AI
       - Demo: Using Gemini for real-time learning
    
    3. Your First 30 Days: Productivity Focus
       - Essential AI tools (Claude, GPT, Gemini)
       - 80/20 rule for AI mastery
       - AI-powered productivity systems
    
    4. Building Your AI Toolkit
       - No-code tools and Zapier AI
       - Understanding APIs simply
       - Connecting tools together
    
    5. From User to Creator
       - Building text and voice agents
       - Real demo: Creating an SMS bot for BTC price tracking
       - Autonomous agent development
    
    6. Turning Skills Into Income
       - Project-based learning
       - Portfolio building
       - Finding clients
    
    7. Scaling Your AI Business
       - Creating systems
       - Building a SaaS
       - Managing AI teams
    
    Target Audience: Beginners looking to build an AI-powered business in 2025
    """
    
    # Generate the intro
    result = crew.generate_intro(sample_outline)
    print("\nIntro has been generated and saved to outputs/generated_intro.md")

if __name__ == "__main__":
    main()
