from src.crew import YouTubeIntroCrew

# Test video outline
outline = '''
Title: "How I Built a 24/7 Sales Team Using AI (No Humans Required)"

Key Sections:
1. The Problem with Traditional Sales Teams
   - High costs and commissions
   - Inconsistent performance
   - Human limitations

2. The AI Solution
   - Multi-channel engagement
   - 24/7 operation
   - Scalability without headaches

3. Real Results
   - Case study: Dead lead revival
   - Revenue numbers
   - Cost savings

Target Audience: Business owners struggling with sales team management
'''

# Initialize the crew and generate intro
crew = YouTubeIntroCrew()
intro = crew.generate_intro(outline)
print("\nGenerated Intro:\n", intro)
