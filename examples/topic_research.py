"""Example: Using the deep research system for general topic research.

This example demonstrates how to research any topic and extract
structured information about it.
"""

import asyncio

from src.graph import graph


async def main():
    """Run topic research on Quantum Computing."""

    # Input for the research graph
    input_data = {
        "research_subject": "Quantum Computing",
        "research_type": "topic",  # Specify the research type
    }

    print(f"Starting topic research on: {input_data['research_subject']}")
    print("=" * 60)

    # Run the research graph
    result = await graph.ainvoke(input_data)

    # The result contains structured_output with topic sections
    structured_output = result.get("structured_output")

    print("\n" + "=" * 60)
    print("Research Complete!")
    print("=" * 60)

    # Print the topic report
    if structured_output:
        print(f"\nTopic: {structured_output.topic_name}")
        print(f"\nSections found: {len(structured_output.sections)}")

        # Print each section
        for section in structured_output.sections:
            print(f"\n{section.category.upper()}: {section.title}")
            print(f"{section.content[:200]}...")
            if section.key_points:
                print(f"\nKey Points:")
                for point in section.key_points:
                    print(f"  • {point}")


if __name__ == "__main__":
    asyncio.run(main())
