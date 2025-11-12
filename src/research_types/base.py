"""Base class and protocol for all research types.

This module defines the abstract interface that all research types must implement.
Each research type encapsulates:
- Prompts for the supervisor, event summarization, and output structuring
- Output schema definitions
- Initial data structures
- Custom output structuring logic
"""

from abc import ABC, abstractmethod
from typing import Any, Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel


class ResearchType(ABC):
    """Abstract base class for research types.

    Each research type must implement all abstract methods to define:
    - How research is prompted and guided
    - What the output schema looks like
    - How data is accumulated and structured

    This allows the core graph to work with any research domain
    without modification.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this research type.

        Returns:
            A lowercase string identifier (e.g., 'biography', 'company', 'market')
        """
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name for this research type.

        Returns:
            A formatted string (e.g., 'Biography', 'Company Research')
        """
        pass

    @abstractmethod
    def get_supervisor_prompt(self) -> str:
        """Returns the supervisor system prompt template.

        The prompt should include placeholders for:
        - {research_subject}: The subject being researched
        - {data_summary}: Current gaps/summary of research
        - {last_message}: The last message in the conversation

        Returns:
            A string template for the supervisor agent
        """
        pass

    @abstractmethod
    def get_event_summarizer_prompt(self) -> str:
        """Returns prompt for analyzing gaps in research.

        This prompt analyzes accumulated data and identifies what's missing.
        Should include placeholder for:
        - {existing_data}: The current accumulated research data

        Returns:
            A string template for gap analysis
        """
        pass

    @abstractmethod
    def get_structure_prompt(self) -> str:
        """Returns prompt for structuring final output.

        This prompt converts accumulated text into structured output.
        Should include placeholder for:
        - {existing_data}: The accumulated research data to structure

        Returns:
            A string template for output structuring
        """
        pass

    @abstractmethod
    def get_output_schema(self) -> Type[BaseModel]:
        """Returns Pydantic schema for final structured output.

        This defines the structure of the final research result.

        Returns:
            A Pydantic BaseModel class
        """
        pass

    @abstractmethod
    def get_initial_data_structure(self) -> Any:
        """Returns initial structure for accumulating research data.

        This defines what the 'existing_data' field starts as.
        Could be a dict, string, list, or Pydantic model.

        Returns:
            Initial data structure (any type)
        """
        pass

    @abstractmethod
    async def structure_output(
        self, existing_data: Any, config: RunnableConfig
    ) -> dict:
        """Custom logic for structuring the final output.

        This method is called at the end of research to convert
        accumulated data into the final structured format.

        Args:
            existing_data: The accumulated research data
            config: Runtime configuration with model settings

        Returns:
            Dictionary containing the final structured output
        """
        pass

    def get_additional_tools(self) -> list:
        """Optional: Returns research-type-specific tools.

        Override this to provide custom tools beyond the standard
        ResearchEventsTool, FinishResearchTool, and think_tool.

        Returns:
            List of tool classes (Pydantic models)
        """
        return []

    def get_subject_display_name(self) -> str:
        """Returns what to call the research subject.

        Examples: 'person', 'company', 'market', 'topic'

        Returns:
            Singular noun for the subject type
        """
        return "subject"
