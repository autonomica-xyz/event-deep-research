"""Registry for discovering and accessing research types.

This module provides a central registry where all research types are registered
and can be retrieved by name. This allows the system to dynamically support
multiple research types without hardcoding them in the core graph logic.
"""

from typing import Dict

from src.research_types.base import ResearchType


class ResearchTypeRegistry:
    """Central registry for all available research types.

    Research types must register themselves to be discoverable by the system.
    The registry provides lookup and enumeration capabilities.
    """

    _types: Dict[str, ResearchType] = {}

    @classmethod
    def register(cls, research_type: ResearchType) -> None:
        """Register a research type with the registry.

        Args:
            research_type: An instance of a ResearchType implementation

        Raises:
            ValueError: If a research type with this name already exists
        """
        if research_type.name in cls._types:
            raise ValueError(
                f"Research type '{research_type.name}' is already registered"
            )
        cls._types[research_type.name] = research_type
        print(f"Registered research type: {research_type.name}")

    @classmethod
    def get(cls, name: str) -> ResearchType:
        """Retrieve a research type by name.

        Args:
            name: The unique identifier for the research type

        Returns:
            The ResearchType instance

        Raises:
            ValueError: If no research type with this name exists
        """
        if name not in cls._types:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"Unknown research type: '{name}'. Available types: {available}"
            )
        return cls._types[name]

    @classmethod
    def list_available(cls) -> list[str]:
        """List all registered research type names.

        Returns:
            List of research type identifiers
        """
        return list(cls._types.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered research types.

        Primarily used for testing.
        """
        cls._types.clear()
