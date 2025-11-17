"""DuckDuckGo search provider implementation.

DuckDuckGo is a privacy-focused search engine with a free API.
No API key required - fully open source option.
"""

from typing import Optional

from src.search_providers.base import SearchProvider, SearchProviderError, SearchResult


class DuckDuckGoSearchProvider(SearchProvider):
    """Search provider using DuckDuckGo.

    This is a free, open-source search option that requires no API key.
    Uses the duckduckgo-search library.
    """

    @property
    def name(self) -> str:
        return "duckduckgo"

    @property
    def requires_api_key(self) -> bool:
        return False

    async def search(
        self,
        query: str,
        max_results: int = 6,
        excluded_domains: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Search using DuckDuckGo.

        Args:
            query: The search query
            max_results: Maximum number of results to return
            excluded_domains: List of domains to exclude

        Returns:
            List of SearchResult objects

        Raises:
            SearchProviderError: If the search fails
        """
        try:
            # Import here to make it optional
            from duckduckgo_search import DDGS

            search_results = []

            # Perform search using context manager
            with DDGS() as ddgs:
                results = ddgs.text(
                    query,
                    max_results=max_results * 2,  # Get extra to filter
                )

                # Convert to list if it's a generator
                if not isinstance(results, list):
                    results = list(results)

                # Filter excluded domains
                for item in results:
                    url = item.get("href", item.get("link", ""))

                    # Check if domain is excluded
                    if excluded_domains:
                        domain_excluded = False
                        for excluded_domain in excluded_domains:
                            if excluded_domain in url:
                                domain_excluded = True
                                break
                        if domain_excluded:
                            continue

                    search_results.append(
                        SearchResult(
                            url=url,
                            title=item.get("title", ""),
                            content=item.get("body", item.get("description", "")),
                            score=None,  # DuckDuckGo doesn't provide scores
                        )
                    )

                    # Stop when we have enough results
                    if len(search_results) >= max_results:
                        break

            return search_results

        except ImportError:
            raise SearchProviderError(
                "DuckDuckGo search requires 'duckduckgo-search' package. "
                "Install with: pip install duckduckgo-search"
            )
        except Exception as e:
            raise SearchProviderError(f"DuckDuckGo search failed: {str(e)}") from e
