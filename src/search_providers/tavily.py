"""Tavily search provider implementation.

Tavily is a commercial search API optimized for AI agents and research tasks.
Requires TAVILY_API_KEY environment variable.
"""

from typing import Optional

from langchain_tavily import TavilySearch

from src.search_providers.base import SearchProvider, SearchProviderError, SearchResult


class TavilySearchProvider(SearchProvider):
    """Search provider using Tavily API.

    Tavily provides high-quality search results optimized for research tasks.
    It's the original search provider used in this system.
    """

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def requires_api_key(self) -> bool:
        return True

    def get_required_env_vars(self) -> list[str]:
        return ["TAVILY_API_KEY"]

    async def search(
        self,
        query: str,
        max_results: int = 6,
        excluded_domains: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Search using Tavily API.

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
            # Create Tavily search tool
            tool = TavilySearch(
                max_results=max_results,
                topic="general",
                include_raw_content=False,
                include_answer=False,
                exclude_domains=excluded_domains or [],
            )

            # Perform search
            result = tool.invoke({"query": query})

            # Convert to our SearchResult format
            search_results = []
            for item in result.get("results", []):
                search_results.append(
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        content=item.get("content", ""),
                        score=item.get("score"),
                    )
                )

            return search_results

        except Exception as e:
            raise SearchProviderError(f"Tavily search failed: {str(e)}") from e
