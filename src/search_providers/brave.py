"""Brave Search API provider implementation.

Brave Search offers a free tier with 2,000 queries per month.
Requires BRAVE_API_KEY environment variable.
"""

import os
from typing import Optional

import aiohttp

from src.search_providers.base import SearchProvider, SearchProviderError, SearchResult


class BraveSearchProvider(SearchProvider):
    """Search provider using Brave Search API.

    Brave Search offers:
    - Free tier: 2,000 queries/month
    - No rate limiting on free tier
    - Privacy-focused results
    - Independent index (not a meta-search)

    Sign up: https://brave.com/search/api/
    """

    @property
    def name(self) -> str:
        return "brave"

    @property
    def requires_api_key(self) -> bool:
        return True

    def get_required_env_vars(self) -> list[str]:
        return ["BRAVE_API_KEY"]

    async def search(
        self,
        query: str,
        max_results: int = 6,
        excluded_domains: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Search using Brave Search API.

        Args:
            query: The search query
            max_results: Maximum number of results to return
            excluded_domains: List of domains to exclude

        Returns:
            List of SearchResult objects

        Raises:
            SearchProviderError: If the search fails
        """
        api_key = os.getenv("BRAVE_API_KEY")
        if not api_key:
            raise SearchProviderError("BRAVE_API_KEY environment variable not set")

        try:
            url = "https://api.search.brave.com/res/v1/web/search"

            headers = {
                "Accept": "application/json",
                "Accept-Encoding": "gzip",
                "X-Subscription-Token": api_key,
            }

            params = {
                "q": query,
                "count": max_results,
                "safesearch": "moderate",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, params=params) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise SearchProviderError(
                            f"Brave API returned status {response.status}: {error_text}"
                        )

                    data = await response.json()

            # Parse results
            search_results = []
            for item in data.get("web", {}).get("results", []):
                url = item.get("url", "")

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
                        content=item.get("description", ""),
                        score=None,  # Brave doesn't provide relevance scores
                    )
                )

            return search_results[:max_results]

        except aiohttp.ClientError as e:
            raise SearchProviderError(f"Brave Search API request failed: {str(e)}") from e
        except Exception as e:
            raise SearchProviderError(f"Brave search failed: {str(e)}") from e
