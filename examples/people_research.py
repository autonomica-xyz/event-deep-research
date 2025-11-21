"""Example: Using the deep research system for people research.

This example demonstrates how to research an individual and extract
a structured profile with demographic, professional, relationship,
achievement, and other relevant information.
"""

import asyncio

from src.graph import graph


async def main():
    """Run people research on Elon Musk."""

    # Input for the research graph
    input_data = {
        "research_subject": "Elon Musk",
        "research_type": "people",  # Specify the research type
    }

    print(f"Starting people research on: {input_data['research_subject']}")
    print("=" * 60)

    # Run the research graph
    result = await graph.ainvoke(input_data)

    # The result contains structured_output with the people profile
    structured_output = result.get("structured_output")

    if not structured_output:
        print("\nNo research results found.")
        return

    print("\n" + "=" * 60)
    print(f"Research Complete! Profile for {structured_output.person_name}")
    print("=" * 60)

    # Print the summary
    print(f"\nSummary:\n{structured_output.summary}")

    # Print facts grouped by category
    print(f"\nTotal Facts: {len(structured_output.facts)}")
    print("\n" + "=" * 60)

    # Group facts by category
    facts_by_category = {}
    for fact in structured_output.facts:
        if fact.category not in facts_by_category:
            facts_by_category[fact.category] = []
        facts_by_category[fact.category].append(fact)

    # Print facts by category
    for category, facts in facts_by_category.items():
        print(f"\n{category.upper()} ({len(facts)} facts):")
        print("-" * 60)
        for i, fact in enumerate(facts[:3], 1):  # Show first 3 per category
            print(f"\n  {i}. {fact.title}")
            if fact.source_date:
                print(f"     Date: {fact.source_date}")
            print(f"     {fact.content[:150]}...")

        if len(facts) > 3:
            print(f"\n  ... and {len(facts) - 3} more {category} facts")


if __name__ == "__main__":
    asyncio.run(main())
