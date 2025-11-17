"""Example: Using DuckDuckGo as your search provider (100% FREE, NO API KEY).

This example demonstrates how to use the DuckDuckGo search provider,
which is completely free and requires no API key.

Perfect for:
- Development and testing
- Personal projects
- When you want to avoid API costs
- Privacy-focused search
"""

import asyncio

from src.graph import graph


async def main():
    """Run research using DuckDuckGo search provider."""

    # Input for the research graph
    input_data = {
        "research_subject": "Marie Curie",
        "research_type": "biography",
        # No API keys needed!
    }

    # Configure to use DuckDuckGo in the runtime config
    config = {
        "configurable": {
            "search_provider": "duckduckgo",  # Use DuckDuckGo
        }
    }

    print("Starting research with DuckDuckGo (free, no API key required)...")
    print("=" * 60)

    # Run the research graph with DuckDuckGo
    result = await graph.ainvoke(input_data, config=config)

    # The result contains structured_output
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


if __name__ == "__main__":
    print("=" * 60)
    print("DuckDuckGo Search Provider Example")
    print("=" * 60)
    print()
    print("✓ No API key required")
    print("✓ Completely free")
    print("✓ Privacy-focused")
    print("✓ Perfect for development and testing")
    print()
    print("Note: DuckDuckGo is auto-selected if no API keys are configured.")
    print()

    asyncio.run(main())
