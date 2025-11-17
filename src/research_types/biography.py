"""Biography research type implementation.

This module implements research for biographical subjects (historical figures, people).
It extracts chronological life events categorized by life phases:
- Early life (childhood, education, influences)
- Personal life (relationships, family, residences)
- Career (professional journey, achievements)
- Legacy (recognition, impact, influence)
"""

from typing import Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.llm_service import create_llm_structured_model
from src.research_types.base import ResearchType


# Biography-specific data models
class CategoriesWithEvents(BaseModel):
    """Biographical events organized by life phases."""

    early: str = Field(
        default="",
        description="Covers childhood, upbringing, family, education, and early influences that shaped the person.",
    )
    personal: str = Field(
        default="",
        description="Focuses on relationships, friendships, family life, places of residence, and notable personal traits or beliefs.",
    )
    career: str = Field(
        default="",
        description="Details their professional journey: first steps, major achievements, collaborations, and significant milestones.",
    )
    legacy: str = Field(
        default="",
        description="Explains how their work was received, awards or recognition, cultural impact, influence on others, and how they are remembered today.",
    )


class ChronologyDate(BaseModel):
    """A structured representation of a date for a chronological event."""

    year: int | None = Field(None, description="The year of the event.")
    note: str | None = Field(
        None, description="Adds extra information to the date (month, day, range...)."
    )


class ChronologyEvent(BaseModel):
    """A single biographical event with full metadata."""

    id: str = Field(
        description="The id of the event in lowercase and underscores. Ex: 'word1_word2'"
    )
    name: str = Field(description="A short, title-like name for the event.")
    description: str = Field(description="A concise description of the event.")
    date: ChronologyDate = Field(..., description="The structured date of the event.")
    location: str | None = Field(
        None, description="The geographical location where the event occurred."
    )


class Chronology(BaseModel):
    """A complete chronological timeline of biographical events."""

    events: list[ChronologyEvent]


class BiographyResearchType(ResearchType):
    """Research type for biographical subjects (people, historical figures)."""

    @property
    def name(self) -> str:
        return "biography"

    @property
    def display_name(self) -> str:
        return "Biography"

    def get_subject_display_name(self) -> str:
        return "person"

    def get_supervisor_prompt(self) -> str:
        return """
You are a meticulous research agent. Your primary directive is to follow a strict, state-based execution cycle to build a comprehensive event timeline for: **{research_subject}**.

<Core Execution Cycle>**
On every turn, you MUST follow these steps in order:

1.  **Step 1: Check for Completion.**
    *   Examine the `<Events Missing>`. If it explicitly states the research is COMPLETE, you MUST immediately call the `FinishResearchTool` and stop.
</Core Execution Cycle>

**CRITICAL CONSTRAINTS:**
*   NEVER call `ResearchEventsTool` twice in a row.
*   NEVER call `think_tool` twice in a row.
*   ALWAYS call exactly ONE tool per turn.

<Events Missing>
{data_summary}
</Events Missing>

<Last Message>
{last_message}
</Last Message>


<Available Tools>

**<Available Tools>**
*   `ResearchEventsTool`: Finds events about the historical figure.
*   `FinishResearchTool`: Ends the research process. Call this ONLY when the research is complete
*   `think_tool`:  Use this to analyze results and plan the EXACT search query for your next action.

**CRITICAL: Use think_tool before calling ResearchEventsTool to plan your approach, and after each ResearchEventsTool to assess progress. Do not call think_tool two times in a row.**
</Available Tools>

1. **Top Priority Gap:** Identify the SINGLE most important missing piece of information from the `<Events Missing>`.
2  **Planned Query:** Write the EXACT search query you will use in the next `ResearchEventsTool` call to fill that gap.

**CRITICAL:** Execute ONLY ONE tool call now, following the `<Core Execution Cycle>`.
"""

    def get_event_summarizer_prompt(self) -> str:
        return """
Analyze the following events and identify only the 2 biggest gaps in information. Be brief and general.

**Events:**
{existing_data}

<Example Gaps:**
- Missing details about Y Time Period in his/her life
</Example Gaps>

**Gaps:**
"""

    def get_structure_prompt(self) -> str:
        return """You are a data processing specialist. Your sole task is to convert a pre-cleaned, chronologically ordered list of life events into a structured JSON object.

<Task>
You will be given a list of events that is already de-duplicated and ordered. You must not change the order or content of the events. For each event in the list, you will extract its name, a detailed description, its date, and location, and format it as JSON.
</Task>

<Guidelines>
1.  For the `name` field, create a short, descriptive title for the event (e.g., "Birth of Pablo Picasso").
2.  For the `description` field, provide the clear and concise summary of what happened from the input text.
3.  For the `date` field, populate `year`, `month`, and `day` whenever possible.
4.  If the date is an estimate or a range (e.g., "circa 1912" or "Between 1920-1924"), you MUST capture that specific text in the `note` field of the date object, and provide your best estimate for the `year`.
5. For the `location` field, populate the location of the event, leave blank if not mentioned
</Guidelines>

<Chronological Events List>
----
{existing_data}
----
</Chronological Events List>

CRITICAL: You must only return the structured JSON output. Do not add any commentary, greetings, or explanations before or after the JSON.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        return Chronology

    def get_initial_data_structure(self) -> CategoriesWithEvents:
        return CategoriesWithEvents(early="", personal="", career="", legacy="")

    async def structure_output(
        self, existing_data: CategoriesWithEvents, config: RunnableConfig
    ) -> dict:
        """Structures biographical events into a chronological timeline.

        Args:
            existing_data: CategoriesWithEvents containing accumulated biographical text
            config: Runtime configuration

        Returns:
            Dictionary with 'structured_output' key containing list of ChronologyEvent objects
        """
        print("--- Structuring Biography Events into JSON ---")

        if not existing_data:
            print("Warning: No events found in state")
            return {"structured_output": []}

        structured_llm = create_llm_structured_model(
            config=config, class_name=Chronology
        )

        # Process each category separately
        early_prompt = self.get_structure_prompt().format(
            existing_data=existing_data.early
        )
        career_prompt = self.get_structure_prompt().format(
            existing_data=existing_data.career
        )
        personal_prompt = self.get_structure_prompt().format(
            existing_data=existing_data.personal
        )
        legacy_prompt = self.get_structure_prompt().format(
            existing_data=existing_data.legacy
        )

        # Invoke LLM for each category
        early_response = await structured_llm.ainvoke(early_prompt)
        career_response = await structured_llm.ainvoke(career_prompt)
        personal_response = await structured_llm.ainvoke(personal_prompt)
        legacy_response = await structured_llm.ainvoke(legacy_prompt)

        # Combine all events
        all_events = (
            early_response.events
            + career_response.events
            + personal_response.events
            + legacy_response.events
        )

        return {"structured_output": all_events}
