"""Example: Using SearXNG as your search provider.

This example demonstrates how to use a SearXNG instance for search.
SearXNG is a free, self-hosted meta-search engine.

SearXNG Benefits:
- Self-hosted: Full control and unlimited queries
- Meta-search: Aggregates results from multiple engines
- Privacy-focused: No tracking
- Public instances available

Setup Options:
1. Self-hosted: https://docs.searxng.org/admin/installation.html
2. Public instances: https://searx.space/
"""

import asyncio
import os

from src.graph import graph


async def main():
    """Run research using SearXNG instance."""

    # Check if instance URL is set
    if not os.getenv("SEARXNG_URL"):
        print("ERROR: SEARXNG_URL environment variable not set!")
        print()
        print("To use SearXNG:")
        print()
        print("Option 1 - Use a public instance:")
        print("  1. Visit https://searx.space/ to find a public instance")
        print("  2. Add to your .env file:")
        print("     SEARXNG_URL=https://searx.be")
        print()
        print("Option 2 - Self-host (Docker):")
        print("  docker run -d -p 8888:8080 searxng/searxng")
        print("  Then add to .env:")
        print("     SEARXNG_URL=http://localhost:8888")
        print()
        print("Option 3 - Self-host (Full install):")
        print("  Follow: https://docs.searxng.org/admin/installation.html")
        return

    # Input for the research graph
    input_data = {
        "research_subject": "Alan Turing",
        "research_type": "biography",
    }

    # Configure to use SearXNG
    config = {
        "configurable": {
            "search_provider": "searxng",  # Use SearXNG
        }
    }

    searxng_url = os.getenv("SEARXNG_URL")
    print(f"Starting research with SearXNG instance at {searxng_url}...")
    print("=" * 60)

    # Run the research graph with SearXNG
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
    print("SearXNG Meta-Search Example")
    print("=" * 60)
    print()
    print("✓ Self-hosted or use public instances")
    print("✓ Meta-search across multiple engines")
    print("✓ Privacy-focused (no tracking)")
    print("✓ Unlimited queries (self-hosted)")
    print()

    asyncio.run(main())
