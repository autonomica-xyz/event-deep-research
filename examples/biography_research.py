"""Example: Using the deep research system for biographical research.

This example demonstrates how to research a historical figure and extract
a structured timeline of their life events.
"""

import asyncio

from src.graph import graph


async def main():
    """Run biographical research on Albert Einstein."""

    # Input for the research graph
    input_data = {
        "research_subject": "Albert Einstein",
        "research_type": "biography",  # Specify the research type
    }

    print(f"Starting biographical research on: {input_data['research_subject']}")
    print("=" * 60)

    # Run the research graph
    result = await graph.ainvoke(input_data)

    # The result contains structured_output with chronological events
    structured_output = result.get("structured_output", [])

    print("\n" + "=" * 60)
    print(f"Research Complete! Found {len(structured_output)} events.")
    print("=" * 60)

    # Print some example events
    if structured_output:
        print("\nSample Events:")
        for i, event in enumerate(structured_output[:5], 1):
            print(f"\n{i}. {event.name}")
            print(f"   Date: {event.date.year}", end="")
            if event.date.note:
                print(f" ({event.date.note})", end="")
            print()
            if event.location:
                print(f"   Location: {event.location}")
            print(f"   Description: {event.description[:100]}...")

        if len(structured_output) > 5:
            print(f"\n... and {len(structured_output) - 5} more events")


if __name__ == "__main__":
    asyncio.run(main())
