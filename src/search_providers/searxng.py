"""SearXNG search provider implementation.

SearXNG is a free, self-hosted meta-search engine that aggregates results
from multiple search engines while protecting privacy.

Requires SEARXNG_URL environment variable pointing to your instance.
"""

import os
from typing import Optional

import aiohttp

from src.search_providers.base import SearchProvider, SearchProviderError, SearchResult


class SearXNGSearchProvider(SearchProvider):
    """Search provider using SearXNG instance.

    SearXNG is a privacy-respecting meta-search engine that can be:
    - Self-hosted: Full control and unlimited queries
    - Use public instances: Free but may have rate limits

    Setup:
    1. Self-hosted: https://docs.searxng.org/admin/installation.html
    2. Public instances: https://searx.space/

    Set SEARXNG_URL to your instance URL (e.g., http://localhost:8888 or https://searx.be)
    """

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def requires_api_key(self) -> bool:
        return False  # API key not required, but instance URL is

    def get_required_env_vars(self) -> list[str]:
        return ["SEARXNG_URL"]

    async def search(
        self,
        query: str,
        max_results: int = 6,
        excluded_domains: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Search using SearXNG instance.

        Args:
            query: The search query
            max_results: Maximum number of results to return
            excluded_domains: List of domains to exclude

        Returns:
            List of SearchResult objects

        Raises:
            SearchProviderError: If the search fails
        """
        searxng_url = os.getenv("SEARXNG_URL")
        if not searxng_url:
            raise SearchProviderError(
                "SEARXNG_URL environment variable not set. "
                "Set it to your SearXNG instance URL (e.g., http://localhost:8888)"
            )

        # Remove trailing slash
        searxng_url = searxng_url.rstrip("/")

        try:
            search_url = f"{searxng_url}/search"

            params = {
                "q": query,
                "format": "json",
                "categories": "general",
            }

            headers = {
                "Accept": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    search_url, params=params, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise SearchProviderError(
                            f"SearXNG returned status {response.status}: {error_text}"
                        )

                    data = await response.json()

            # Parse results
            search_results = []
            for item in data.get("results", []):
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
                        content=item.get("content", ""),
                        score=item.get("score"),  # SearXNG provides scores
                    )
                )

                # Stop when we have enough results
                if len(search_results) >= max_results:
                    break

            return search_results

        except aiohttp.ClientError as e:
            raise SearchProviderError(
                f"SearXNG request failed. Is your instance URL correct? Error: {str(e)}"
            ) from e
        except Exception as e:
            raise SearchProviderError(f"SearXNG search failed: {str(e)}") from e
