"""Example: Using the deep research system for company research.

This example demonstrates how to research a company and extract
structured information about it.
"""

import asyncio

from src.graph import graph


async def main():
    """Run company research on OpenAI."""

    # Input for the research graph
    input_data = {
        "research_subject": "OpenAI",
        "research_type": "company",  # Specify the research type
    }

    print(f"Starting company research on: {input_data['research_subject']}")
    print("=" * 60)

    # Run the research graph
    result = await graph.ainvoke(input_data)

    # The result contains structured_output with company facts
    structured_output = result.get("structured_output")

    print("\n" + "=" * 60)
    print("Research Complete!")
    print("=" * 60)

    # Print the company profile
    if structured_output:
        print(f"\nCompany: {structured_output.company_name}")
        print(f"\nFacts found: {len(structured_output.facts)}")

        # Group by category
        categories = {}
        for fact in structured_output.facts:
            if fact.category not in categories:
                categories[fact.category] = []
            categories[fact.category].append(fact)

        # Print facts by category
        for category, facts in categories.items():
            print(f"\n{category.upper()}:")
            for fact in facts:
                print(f"  • {fact.title}")
                print(f"    {fact.content[:150]}...")
                if fact.source_date:
                    print(f"    (As of: {fact.source_date})")


if __name__ == "__main__":
    asyncio.run(main())
