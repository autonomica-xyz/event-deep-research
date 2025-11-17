"""Defines the Pydantic models and TypedDicts for the research agent graph.
This file serves as the schema for data structures, agent tools, and state management.

The state models have been generalized to support multiple research types (biography, company, market, etc.).
Research-type-specific models are defined in src/research_types/*.py files.
"""

import operator
from typing import Annotated, Any, List, TypedDict

from langchain_core.messages import MessageLikeRepresentation
from pydantic import BaseModel, Field

################################################################################
# Section 1: Core Data Models (DEPRECATED - kept for backward compatibility)
# - These biography-specific models are now defined in src/research_types/biography.py
# - They remain here for backward compatibility with existing code
################################################################################


class ChronologyDate(BaseModel):
    """A structured representation of a date for a chronological event.

    DEPRECATED: This is now defined in src/research_types/biography.py
    Kept here for backward compatibility.
    """

    year: int | None = Field(None, description="The year of the event.")
    note: str | None = Field(
        None, description="Adds extra information to the date (month, day, range...)."
    )


class ChronologyEventInput(BaseModel):
    """Represents a single event, typically used for initial data extraction before an ID is assigned.

    DEPRECATED: This is now defined in src/research_types/biography.py
    Kept here for backward compatibility.
    """

    name: str = Field(description="A short, title-like name for the event.")
    description: str = Field(description="A concise description of the event.")
    date: ChronologyDate = Field(..., description="The structured date of the event.")
    location: str | None = Field(
        None, description="The geographical location where the event occurred."
    )


class ChronologyEvent(ChronologyEventInput):
    """The final, canonical event model with a unique identifier.

    DEPRECATED: This is now defined in src/research_types/biography.py
    Kept here for backward compatibility.
    """

    id: str = Field(
        description="The id of the event in lowercase and underscores. Ex: 'word1_word2'"
    )


class ChronologyInput(BaseModel):
    """A list of newly extracted events from a research source.

    DEPRECATED: This is now defined in src/research_types/biography.py
    Kept here for backward compatibility.
    """

    events: list[ChronologyEventInput]


class Chronology(BaseModel):
    """A complete chronological timeline with finalized (ID'd) events.

    DEPRECATED: This is now defined in src/research_types/biography.py
    Kept here for backward compatibility.
    """

    events: list[ChronologyEvent]


class CategoriesWithEvents(BaseModel):
    """Biographical events organized by life phases.

    DEPRECATED: This is now defined in src/research_types/biography.py
    Kept here for backward compatibility.
    """

    early: str = Field(
        default="",
        description="Covers childhood, upbringing, family, education, and early influences that shaped the author.",
    )
    personal: str = Field(
        default="",
        description="Focuses on relationships, friendships, family life, places of residence, and notable personal traits or beliefs.",
    )
    career: str = Field(
        default="",
        description="Details their professional journey: first steps into writing, major publications, collaborations, recurring themes, style, and significant milestones.",
    )
    legacy: str = Field(
        default="",
        description="Explains how their work was received, awards or recognition, cultural/literary impact, influence on other authors, and how they are remembered today.",
    )


################################################################################
# Section 2: Agent Tools
# - Pydantic models that define the tools available to the LLM agents.
################################################################################


class ResearchEventsTool(BaseModel):
    """The query to be used to research events about an historical figure. The query is based on the reflection of the assistant."""

    research_question: str
    pass  # No arguments needed


class FinishResearchTool(BaseModel):
    """Concludes the research process.
    Call this tool ONLY when you have a comprehensive timeline of the person's life,
    including key events like birth, death, major achievements, and significant personal
    milestones, and you are confident that no major gaps remain.
    """

    pass


################################################################################
# Section 3: Graph State Definitions
# - TypedDicts and models that define the "memory" for the agent graphs.
################################################################################


def override_reducer(current_value, new_value):
    """Reducer function that allows a new value to completely replace the old one."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


# --- Main Supervisor Graph State (Generalized for all research types) ---


class SupervisorStateInput(TypedDict):
    """The initial input to start the main research graph.

    This state is now generic and works with any research type.
    """

    research_subject: str  # Generic: can be person, company, market, topic, etc.
    research_type: str  # Type of research: 'biography', 'company', 'market', 'topic'
    existing_data: Any = Field(
        default=None,
        description="Accumulated research data (structure depends on research type).",
    )
    used_domains: list[str] = Field(
        default=[],
        description="The domains that have been used to extract information.",
    )
    data_summary: str = Field(
        default="",
        description="A summary of gaps in the current research data.",
    )


class SupervisorState(SupervisorStateInput):
    """The complete state for the main supervisor graph.

    This state works with any research type through the research_type field.
    """

    conversation_history: Annotated[list[MessageLikeRepresentation], override_reducer]
    iteration_count: int = 0
    structured_output: Any | None = Field(
        default=None,
        description="Final structured output (type depends on research type).",
    )


# --- Backward Compatibility Aliases ---
# These allow old code using person_to_research to still work

class SupervisorStateInputLegacy(TypedDict):
    """DEPRECATED: Use SupervisorStateInput with research_subject instead.

    This is kept for backward compatibility with existing tests and code.
    """

    person_to_research: str
    existing_events: CategoriesWithEvents = Field(
        default=CategoriesWithEvents(early="", personal="", career="", legacy=""),
        description="Covers chronology events of the person to research.",
    )
    used_domains: list[str] = Field(
        default=[],
        description="The domains that have been used to extract events.",
    )
    events_summary: str = Field(
        default="",
        description="A summary of the events.",
    )


class SupervisorStateLegacy(SupervisorStateInputLegacy):
    """DEPRECATED: Use SupervisorState with research_type instead.

    This is kept for backward compatibility with existing tests and code.
    """

    final_events: List[ChronologyEvent]
    conversation_history: Annotated[list[MessageLikeRepresentation], override_reducer]
    iteration_count: int = 0
    structured_events: list[ChronologyEvent] | None
