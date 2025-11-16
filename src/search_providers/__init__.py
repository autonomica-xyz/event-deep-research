"""Search providers module.

This module provides a pluggable search provider system that allows the deep research
agent to work with different search backends:

Available Providers:
- **Tavily** (tavily): Commercial API, high quality results (requires API key)
- **Brave** (brave): Free tier 2000 queries/month (requires API key)
- **DuckDuckGo** (duckduckgo): Free, no API key needed (open source)
- **SearXNG** (searxng): Self-hosted meta-search (no API key, needs instance URL)

To add a new search provider:
1. Create a new file in this directory (e.g., my_search.py)
2. Implement a class that inherits from SearchProvider
3. Register it in this __init__.py file

Example:
    from src.search_providers import SearchProviderRegistry

    # Get the default provider
    provider = SearchProviderRegistry.get()

    # Or get a specific provider
    provider = SearchProviderRegistry.get("brave")

    # Search
    results = await provider.search("quantum computing", max_results=10)
"""

from src.search_providers.base import SearchProvider, SearchResult
from src.search_providers.brave import BraveSearchProvider
from src.search_providers.duckduckgo import DuckDuckGoSearchProvider
from src.search_providers.registry import SearchProviderRegistry
from src.search_providers.searxng import SearXNGSearchProvider
from src.search_providers.tavily import TavilySearchProvider

# Register all available search providers
# Tavily is registered first and set as default (original behavior)
try:
    SearchProviderRegistry.register(TavilySearchProvider(), set_as_default=True)
except Exception as e:
    print(f"Note: Tavily provider registration skipped: {e}")

try:
    SearchProviderRegistry.register(BraveSearchProvider())
except Exception as e:
    print(f"Note: Brave provider registration skipped: {e}")

try:
    SearchProviderRegistry.register(DuckDuckGoSearchProvider())
except Exception as e:
    print(f"Note: DuckDuckGo provider registration skipped: {e}")

try:
    SearchProviderRegistry.register(SearXNGSearchProvider())
except Exception as e:
    print(f"Note: SearXNG provider registration skipped: {e}")

# Print available and configured providers on import
print(f"Available search providers: {', '.join(SearchProviderRegistry.list_available())}")
configured = SearchProviderRegistry.list_configured()
if configured:
    print(f"Configured providers: {', '.join(configured)}")
    # Auto-select first configured provider as default if Tavily is not configured
    default = SearchProviderRegistry.get_default()
    if default:
        default_provider = SearchProviderRegistry.get(default)
        is_valid, _ = default_provider.validate_config()
        if not is_valid and configured:
            SearchProviderRegistry.set_default(configured[0])
            print(f"Auto-selected {configured[0]} as default (Tavily not configured)")
else:
    print("Warning: No search providers are properly configured. Set API keys in .env file.")

__all__ = [
    "SearchProvider",
    "SearchResult",
    "SearchProviderRegistry",
    "TavilySearchProvider",
    "BraveSearchProvider",
    "DuckDuckGoSearchProvider",
    "SearXNGSearchProvider",
]
