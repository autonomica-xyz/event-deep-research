"""Example: Using Brave Search as your search provider.

This example demonstrates how to use the Brave Search API,
which offers a generous free tier of 2,000 queries per month.

Brave Search Benefits:
- Free tier: 2,000 queries/month
- Independent index (not a meta-search)
- Privacy-focused
- No rate limiting on free tier

Get your API key: https://brave.com/search/api/
"""

import asyncio
import os

from src.graph import graph


async def main():
    """Run research using Brave Search API."""

    # Check if API key is set
    if not os.getenv("BRAVE_API_KEY"):
        print("ERROR: BRAVE_API_KEY environment variable not set!")
        print()
        print("To use Brave Search:")
        print("1. Sign up at https://brave.com/search/api/")
        print("2. Get your API key")
        print("3. Add to your .env file:")
        print("   BRAVE_API_KEY=your_api_key_here")
        return

    # Input for the research graph
    input_data = {
        "research_subject": "Ada Lovelace",
        "research_type": "biography",
    }

    # Configure to use Brave Search
    config = {
        "configurable": {
            "search_provider": "brave",  # Use Brave Search
        }
    }

    print("Starting research with Brave Search API...")
    print("=" * 60)

    # Run the research graph with Brave Search
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
    print("Brave Search API Example")
    print("=" * 60)
    print()
    print("✓ Free tier: 2,000 queries/month")
    print("✓ Independent search index")
    print("✓ Privacy-focused")
    print("✓ No rate limiting on free tier")
    print()

    asyncio.run(main())
