"""Example: Search Provider Comparison.

This script helps you compare different search providers and see which ones
are configured and available on your system.
"""

import asyncio

from src.search_providers import SearchProviderRegistry


async def test_search_provider(provider_name: str, query: str = "test query"):
    """Test a single search provider."""
    try:
        provider = SearchProviderRegistry.get(provider_name)

        # Check configuration
        is_valid, error_msg = provider.validate_config()

        if not is_valid:
            print(f"   Status: ❌ NOT CONFIGURED")
            print(f"   Reason: {error_msg}")
            if provider.get_required_env_vars():
                print(f"   Required: {', '.join(provider.get_required_env_vars())}")
            return False

        print(f"   Status: ✓ CONFIGURED")

        # Try a quick search
        print(f"   Testing search...")
        results = await provider.search(query, max_results=3)
        print(f"   Results: Found {len(results)} results")

        if results:
            print(f"   Sample: {results[0].title[:50]}...")

        return True

    except Exception as e:
        print(f"   Status: ❌ ERROR")
        print(f"   Error: {str(e)}")
        return False


async def main():
    """Compare all available search providers."""

    print("=" * 70)
    print("SEARCH PROVIDER COMPARISON")
    print("=" * 70)
    print()

    # Get all providers
    providers = SearchProviderRegistry.list_available()
    configured = SearchProviderRegistry.list_configured()
    default = SearchProviderRegistry.get_default()

    print(f"Available providers: {len(providers)}")
    print(f"Configured providers: {len(configured)}")
    print(f"Default provider: {default or 'None'}")
    print()

    # Test each provider
    for provider_name in providers:
        provider = SearchProviderRegistry.get(provider_name)

        print("-" * 70)
        print(f"Provider: {provider.name.upper()}")
        print(f"   API Key Required: {'Yes' if provider.requires_api_key else 'No'}")

        # Add provider descriptions
        if provider.name == "tavily":
            print("   Description: Commercial API, high-quality results")
            print("   Cost: Paid (subscription required)")
        elif provider.name == "brave":
            print("   Description: Independent search, free tier available")
            print("   Cost: Free tier (2,000 queries/month)")
        elif provider.name == "duckduckgo":
            print("   Description: Privacy-focused, no API key needed")
            print("   Cost: Free (unlimited)")
        elif provider.name == "searxng":
            print("   Description: Self-hosted meta-search")
            print("   Cost: Free (requires instance URL)")

        # Test the provider
        await test_search_provider(provider.name)
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if configured:
        print("✓ Ready to use:")
        for prov_name in configured:
            marker = " (DEFAULT)" if prov_name == default else ""
            print(f"  - {prov_name}{marker}")
    else:
        print("❌ No providers configured!")
        print()
        print("Quick setup:")
        print()
        print("  1. DuckDuckGo (easiest, no setup needed):")
        print("     Already available - no configuration required!")
        print()
        print("  2. Brave Search (free tier):")
        print("     - Sign up: https://brave.com/search/api/")
        print("     - Add to .env: BRAVE_API_KEY=your_key")
        print()
        print("  3. SearXNG (self-hosted):")
        print("     - Quick start: docker run -d -p 8888:8080 searxng/searxng")
        print("     - Add to .env: SEARXNG_URL=http://localhost:8888")

    print()


if __name__ == "__main__":
    asyncio.run(main())
