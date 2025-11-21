"""Research types module.

This module provides the pluggable research type system that allows the deep research
agent to work with different domains (biography, company, market, topic, etc.).

To add a new research type:
1. Create a new file in this directory (e.g., my_type.py)
2. Implement a class that inherits from ResearchType
3. Register it in this __init__.py file

Example:
    from src.research_types.base import ResearchType
    from src.research_types.registry import ResearchTypeRegistry
    from src.research_types.my_type import MyResearchType

    # Register your type
    ResearchTypeRegistry.register(MyResearchType())
"""

from src.research_types.base import ResearchType
from src.research_types.biography import BiographyResearchType
from src.research_types.company import CompanyResearchType
from src.research_types.market import MarketResearchType
from src.research_types.people import PeopleResearchType
from src.research_types.registry import ResearchTypeRegistry
from src.research_types.topic import TopicResearchType

# Register all available research types
ResearchTypeRegistry.register(BiographyResearchType())
ResearchTypeRegistry.register(CompanyResearchType())
ResearchTypeRegistry.register(MarketResearchType())
ResearchTypeRegistry.register(PeopleResearchType())
ResearchTypeRegistry.register(TopicResearchType())

__all__ = [
    "ResearchType",
    "ResearchTypeRegistry",
    "BiographyResearchType",
    "CompanyResearchType",
    "MarketResearchType",
    "PeopleResearchType",
    "TopicResearchType",
]
