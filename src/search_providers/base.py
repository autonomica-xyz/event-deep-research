"""Base class and protocol for search providers.

This module defines the abstract interface that all search providers must implement.
Search providers are responsible for finding relevant URLs for a research question.
"""

from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    """A single search result."""

    url: str = Field(description="The URL of the search result")
    title: str = Field(description="The title of the page")
    content: str = Field(description="Snippet or description of the content")
    score: Optional[float] = Field(
        default=None, description="Relevance score (if available)"
    )


class SearchProvider(ABC):
    """Abstract base class for search providers.

    Each search provider implements a specific search backend:
    - Tavily (commercial API)
    - SearXNG (self-hosted meta-search)
    - Brave Search (free tier available)
    - DuckDuckGo (free, no API key needed)
    - etc.

    This allows the system to work with any search backend without
    modification to the core research logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this search provider.

        Returns:
            A lowercase string identifier (e.g., 'tavily', 'searxng', 'brave')
        """
        pass

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether this provider requires an API key.

        Returns:
            True if API key is required, False otherwise
        """
        pass

    @abstractmethod
    async def search(
        self,
        query: str,
        max_results: int = 6,
        excluded_domains: Optional[list[str]] = None,
    ) -> list[SearchResult]:
        """Perform a search query.

        Args:
            query: The search query string
            max_results: Maximum number of results to return
            excluded_domains: List of domains to exclude from results

        Returns:
            List of SearchResult objects

        Raises:
            SearchProviderError: If search fails
        """
        pass

    def get_required_env_vars(self) -> list[str]:
        """Get list of required environment variables.

        Override this if your provider needs API keys or configuration.

        Returns:
            List of environment variable names (e.g., ['BRAVE_API_KEY'])
        """
        return []

    def validate_config(self) -> tuple[bool, Optional[str]]:
        """Validate that the provider is properly configured.

        Returns:
            Tuple of (is_valid, error_message)
        """
        import os

        for env_var in self.get_required_env_vars():
            if not os.getenv(env_var):
                return False, f"Missing required environment variable: {env_var}"
        return True, None


class SearchProviderError(Exception):
    """Exception raised when search provider fails."""

    pass
