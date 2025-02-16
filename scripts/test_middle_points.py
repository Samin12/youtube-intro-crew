import argparse
from src.middle_point_crew import MiddlePointCrew
from src.agents.middle_point.example_brick import ExampleType

def main():
    parser = argparse.ArgumentParser(description='Generate middle points for a video script')
    parser.add_argument('--example-type', type=str, choices=['story', 'metaphor', 'framework'],
                      default='story', help='Type of example to use')
    
    args = parser.parse_args()
    
    # Example video points
    video_points = [
        {
            "title": "ChatGPT for Content Creation",
            "key_points": [
                "Automates blog writing and social media posts",
                "Generates engaging headlines and hooks",
                "Helps with content research and outlines"
            ],
            "context": "This tool helps create high-quality content quickly"
        },
        {
            "title": "Midjourney for Visual Design",
            "key_points": [
                "Creates custom images and graphics",
                "Generates brand-consistent visuals",
                "Saves thousands on design costs"
            ],
            "context": "This AI tool revolutionizes design workflow"
        }
    ]
    
    # Create and run the crew
    crew = MiddlePointCrew()
    example_type = ExampleType(args.example_type)
    result = crew.generate_middle_points(video_points, example_type)
    print(result)

if __name__ == "__main__":
    main()
