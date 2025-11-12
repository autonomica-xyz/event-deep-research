"""Topic research type implementation.

This module implements general-purpose topic research.
It's a flexible research type for any subject that doesn't fit
into more specific categories like biography, company, or market.

Research is organized into:
- Overview & Definition
- Key Concepts & Components
- Historical Context & Development
- Current State & Applications
- Challenges & Controversies
- Future Directions
"""

from typing import Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.llm_service import create_llm_structured_model
from src.research_types.base import ResearchType


class TopicData(BaseModel):
    """Topic research data organized by aspect."""

    overview: str = Field(
        default="",
        description="Definition, basic explanation, and general overview of the topic.",
    )
    concepts: str = Field(
        default="",
        description="Key concepts, components, principles, and important terminology.",
    )
    history: str = Field(
        default="",
        description="Historical background, development, evolution, and origins.",
    )
    current: str = Field(
        default="",
        description="Current state, modern applications, recent developments, and use cases.",
    )
    challenges: str = Field(
        default="",
        description="Challenges, controversies, debates, limitations, and criticisms.",
    )
    future: str = Field(
        default="",
        description="Future directions, potential developments, and emerging trends.",
    )


class TopicSection(BaseModel):
    """A section of topic research."""

    category: str = Field(
        description="Category (overview, concepts, history, current, challenges, future)"
    )
    title: str = Field(description="Section title")
    content: str = Field(description="Main content for this section")
    key_points: list[str] = Field(
        default_factory=list, description="Key takeaways from this section"
    )


class TopicReport(BaseModel):
    """Complete topic research report."""

    topic_name: str = Field(description="Name of the topic")
    sections: list[TopicSection] = Field(description="Research sections")


class TopicResearchType(ResearchType):
    """Research type for general topics and subjects."""

    @property
    def name(self) -> str:
        return "topic"

    @property
    def display_name(self) -> str:
        return "Topic Research"

    def get_subject_display_name(self) -> str:
        return "topic"

    def get_supervisor_prompt(self) -> str:
        return """
You are a meticulous research agent. Your primary directive is to build comprehensive research on: **{research_subject}**.

<Core Execution Cycle>
On every turn, you MUST follow these steps in order:

1.  **Step 1: Check for Completion.**
    *   Examine the `<Information Missing>`. If it explicitly states the research is COMPLETE, you MUST immediately call the `FinishResearchTool` and stop.
</Core Execution Cycle>

**CRITICAL CONSTRAINTS:**
*   NEVER call `ResearchEventsTool` twice in a row.
*   NEVER call `think_tool` twice in a row.
*   ALWAYS call exactly ONE tool per turn.

<Information Missing>
{data_summary}
</Information Missing>

<Last Message>
{last_message}
</Last Message>

<Available Tools>
*   `ResearchEventsTool`: Searches for information about the topic.
*   `FinishResearchTool`: Ends the research process. Call this ONLY when the research is complete.
*   `think_tool`:  Use this to analyze results and plan the EXACT search query for your next action.

**CRITICAL: Use think_tool before calling ResearchEventsTool to plan your approach, and after each ResearchEventsTool to assess progress. Do not call think_tool two times in a row.**
</Available Tools>

Focus on gathering:
1. Overview and definition of the topic
2. Key concepts and components
3. Historical context and development
4. Current state and applications
5. Challenges and controversies
6. Future directions and trends

**CRITICAL:** Execute ONLY ONE tool call now, following the `<Core Execution Cycle>`.
"""

    def get_event_summarizer_prompt(self) -> str:
        return """
Analyze the following topic research and identify the 2 biggest gaps. Be brief and specific.

**Topic Information:**
{existing_data}

<Example Gaps>
- Missing historical context and origins
- Missing current applications and use cases
</Example Gaps>

**Gaps:**
"""

    def get_structure_prompt(self) -> str:
        return """You are a research synthesis specialist. Convert the topic research into structured sections.

<Task>
Organize the research data into clear sections. Each section should have:
- category: Which aspect (overview, concepts, history, current, challenges, future)
- title: A descriptive section title
- content: The main content for this section
- key_points: List of key takeaways (3-5 bullet points)
</Task>

<Topic Research Data>
----
{existing_data}
----
</Topic Research Data>

CRITICAL: Return only the structured JSON output. No commentary or explanations.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        return TopicReport

    def get_initial_data_structure(self) -> TopicData:
        return TopicData(
            overview="",
            concepts="",
            history="",
            current="",
            challenges="",
            future="",
        )

    async def structure_output(
        self, existing_data: TopicData, config: RunnableConfig
    ) -> dict:
        """Structures topic research into a report.

        Args:
            existing_data: TopicData containing accumulated research
            config: Runtime configuration

        Returns:
            Dictionary with 'structured_output' key containing TopicReport
        """
        print("--- Structuring Topic Research Report ---")

        if not existing_data:
            print("Warning: No topic data found in state")
            return {"structured_output": None}

        # Combine all topic data
        combined_data = f"""
Overview & Definition:
{existing_data.overview}

Key Concepts:
{existing_data.concepts}

Historical Context:
{existing_data.history}

Current State:
{existing_data.current}

Challenges:
{existing_data.challenges}

Future Directions:
{existing_data.future}
"""

        structured_llm = create_llm_structured_model(
            config=config, class_name=TopicReport
        )

        prompt = self.get_structure_prompt().format(existing_data=combined_data)
        response = await structured_llm.ainvoke(prompt)

        return {"structured_output": response}
