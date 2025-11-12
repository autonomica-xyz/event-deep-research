"""Example: Using the deep research system for market research.

This example demonstrates how to research a market/industry and extract
structured insights about it.
"""

import asyncio

from src.graph import graph


async def main():
    """Run market research on the AI chip market."""

    # Input for the research graph
    input_data = {
        "research_subject": "AI Chip Market",
        "research_type": "market",  # Specify the research type
    }

    print(f"Starting market research on: {input_data['research_subject']}")
    print("=" * 60)

    # Run the research graph
    result = await graph.ainvoke(input_data)

    # The result contains structured_output with market insights
    structured_output = result.get("structured_output")

    print("\n" + "=" * 60)
    print("Research Complete!")
    print("=" * 60)

    # Print the market report
    if structured_output:
        print(f"\nMarket: {structured_output.market_name}")
        print(f"\nInsights found: {len(structured_output.insights)}")

        # Group by category
        categories = {}
        for insight in structured_output.insights:
            if insight.category not in categories:
                categories[insight.category] = []
            categories[insight.category].append(insight)

        # Print insights by category
        for category, insights in categories.items():
            print(f"\n{category.upper()}:")
            for insight in insights:
                print(f"  • {insight.title}")
                print(f"    {insight.insight[:150]}...")
                if insight.data_points:
                    print(f"    Data points:")
                    for dp in insight.data_points[:3]:
                        print(f"      - {dp}")


if __name__ == "__main__":
    asyncio.run(main())
