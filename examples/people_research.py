"""Example: Using the deep research system for people research with BREADTH-FIRST strategy.

This example demonstrates how to research an individual using parallel research
across 7 domains simultaneously:
1. Professional & Social Media (LinkedIn, career)
2. Technical Contributions (GitHub, open source)
3. Cryptocurrency/Blockchain (Bitcoin, crypto projects)
4. Publications & Media (articles, interviews)
5. Business & Legal (companies, startups)
6. Academic & Education (degrees, research)
7. Community & Speaking (conferences, talks)

The breadth-first approach executes all 7 research queries in parallel for
comprehensive coverage, then fills gaps as needed.
"""

import asyncio

from src.graph import graph


async def main():
    """Run breadth-first people research on Elon Musk."""

    # Input for the research graph
    input_data = {
        "research_subject": "Elon Musk",
        "research_type": "people",  # Specify the research type (uses breadth-first)
    }

    print(f"Starting BREADTH-FIRST people research on: {input_data['research_subject']}")
    print("This will execute 7 parallel research queries across different domains...")
    print("=" * 80)

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
    print("\n" + "=" * 80)

    # Group facts by category
    facts_by_category = {}
    for fact in structured_output.facts:
        if fact.category not in facts_by_category:
            facts_by_category[fact.category] = []
        facts_by_category[fact.category].append(fact)

    # Define category order (core categories first, then breadth-first domains)
    category_order = [
        "demographics",
        "professional",
        "relationships",
        "public_presence",
        "achievements",
        "controversies",
        "technical_contributions",
        "crypto_blockchain",
        "business_ventures",
        "academic_background",
        "community_engagement",
    ]

    # Print facts by category in order
    for category in category_order:
        if category not in facts_by_category:
            continue

        facts = facts_by_category[category]
        print(f"\n{category.upper().replace('_', ' ')} ({len(facts)} facts):")
        print("-" * 80)
        for i, fact in enumerate(facts[:3], 1):  # Show first 3 per category
            print(f"\n  {i}. {fact.title}")
            if fact.source_date:
                print(f"     Date: {fact.source_date}")
            print(f"     {fact.content[:150]}...")

        if len(facts) > 3:
            print(f"\n  ... and {len(facts) - 3} more {category.replace('_', ' ')} facts")

    print("\n" + "=" * 80)
    print("Breadth-first research complete!")
    print(f"Researched across {len(facts_by_category)} categories")


if __name__ == "__main__":
    asyncio.run(main())
