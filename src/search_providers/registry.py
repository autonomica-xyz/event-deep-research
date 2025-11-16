"""Registry for discovering and accessing search providers.

This module provides a central registry where all search providers are registered
and can be retrieved by name. This allows the system to dynamically support
multiple search providers without hardcoding them.
"""

from typing import Dict, Optional

from src.search_providers.base import SearchProvider


class SearchProviderRegistry:
    """Central registry for all available search providers.

    Search providers must register themselves to be discoverable by the system.
    The registry provides lookup and enumeration capabilities.
    """

    _providers: Dict[str, SearchProvider] = {}
    _default_provider: Optional[str] = None

    @classmethod
    def register(cls, provider: SearchProvider, set_as_default: bool = False) -> None:
        """Register a search provider with the registry.

        Args:
            provider: An instance of a SearchProvider implementation
            set_as_default: If True, set this as the default provider

        Raises:
            ValueError: If a provider with this name already exists
        """
        if provider.name in cls._providers:
            raise ValueError(
                f"Search provider '{provider.name}' is already registered"
            )
        cls._providers[provider.name] = provider
        print(f"Registered search provider: {provider.name}")

        # Set as default if it's the first provider or explicitly requested
        if set_as_default or cls._default_provider is None:
            cls._default_provider = provider.name
            print(f"Set default search provider: {provider.name}")

    @classmethod
    def get(cls, name: Optional[str] = None) -> SearchProvider:
        """Retrieve a search provider by name.

        Args:
            name: The unique identifier for the search provider.
                  If None, returns the default provider.

        Returns:
            The SearchProvider instance

        Raises:
            ValueError: If no provider with this name exists or no default is set
        """
        # Use default if no name provided
        if name is None:
            if cls._default_provider is None:
                raise ValueError("No default search provider configured")
            name = cls._default_provider

        if name not in cls._providers:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"Unknown search provider: '{name}'. Available providers: {available}"
            )
        return cls._providers[name]

    @classmethod
    def set_default(cls, name: str) -> None:
        """Set the default search provider.

        Args:
            name: The name of the provider to set as default

        Raises:
            ValueError: If no provider with this name exists
        """
        if name not in cls._providers:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"Unknown search provider: '{name}'. Available providers: {available}"
            )
        cls._default_provider = name
        print(f"Set default search provider: {name}")

    @classmethod
    def get_default(cls) -> Optional[str]:
        """Get the name of the default search provider.

        Returns:
            The name of the default provider, or None if not set
        """
        return cls._default_provider

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered search provider names.

        Returns:
            List of search provider identifiers
        """
        return list(cls._providers.keys())

    @classmethod
    def list_configured(cls) -> list[str]:
        """List search providers that are properly configured.

        Returns:
            List of provider names that have valid configuration
        """
        configured = []
        for name, provider in cls._providers.items():
            is_valid, _ = provider.validate_config()
            if is_valid:
                configured.append(name)
        return configured

    @classmethod
    def clear(cls) -> None:
        """Clear all registered search providers.

        Primarily used for testing.
        """
        cls._providers.clear()
        cls._default_provider = None
