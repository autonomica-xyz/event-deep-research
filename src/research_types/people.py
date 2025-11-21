"""People research type for deep research on individuals."""

from typing import Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.llm_service import create_llm_structured_model
from src.research_types.base import ResearchType


# Data structure for accumulating research
class PeopleData(BaseModel):
    """People research organized by information type."""

    demographics: str = Field(
        default="",
        description="Age, nationality, location, ethnicity, background",
    )
    professional: str = Field(
        default="",
        description="Career, skills, accomplishments, expertise, work history",
    )
    relationships: str = Field(
        default="",
        description="Family, colleagues, mentors, partnerships, connections",
    )
    public_presence: str = Field(
        default="",
        description="Social media, publications, interviews, media appearances, visibility",
    )
    achievements: str = Field(
        default="",
        description="Awards, recognition, impact, contributions, milestones",
    )
    controversies: str = Field(
        default="",
        description="Controversies, legal issues, criticisms, challenges",
    )


# Output schema - what the final result looks like
class PeopleFact(BaseModel):
    """A single fact about a person."""

    category: str = Field(
        description="Category (demographics, professional, relationships, public_presence, achievements, controversies)"
    )
    title: str = Field(description="Short title for this fact")
    content: str = Field(description="Detailed information")
    source_date: str | None = Field(
        None, description="When this information was published/updated"
    )


class PeopleProfile(BaseModel):
    """Complete people research profile."""

    person_name: str = Field(description="Person's full name")
    summary: str = Field(description="Brief overview of who this person is")
    facts: list[PeopleFact] = Field(description="List of researched facts")


class PeopleResearchType(ResearchType):
    """Research type for people and individuals."""

    @property
    def name(self) -> str:
        return "people"

    @property
    def display_name(self) -> str:
        return "People Research"

    def get_subject_display_name(self) -> str:
        return "person"

    def get_supervisor_prompt(self) -> str:
        return """
You are a meticulous people research agent. Your primary directive is to build a comprehensive profile for: **{research_subject}**.

<Core Execution Cycle>
On every turn, you MUST follow these steps in order:

1.  **Step 1: Check for Completion.**
    *   Examine the `<Information Missing>`. If it explicitly states the research is COMPLETE, you MUST immediately call the `FinishResearchTool` and stop.
    *   If gaps remain, proceed to Step 2.

2.  **Step 2: Analyze and Plan (if needed).**
    *   Review `<Last Message>` to understand what just happened.
    *   If you just called `ResearchEventsTool`, you MUST call `think_tool` to assess what was learned and plan your next move.
    *   If you just called `think_tool`, you MUST call `ResearchEventsTool` with the search query you planned.
    *   Use `think_tool` to formulate your EXACT search query before calling `ResearchEventsTool`.

3.  **Step 3: Execute Research.**
    *   Call `ResearchEventsTool` with your planned query to gather information.
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
*   `ResearchEventsTool`: Searches for information about the person. Takes a research_question parameter.
*   `FinishResearchTool`: Ends the research process. Call this ONLY when the research is complete.
*   `think_tool`: Use this to analyze results and plan the EXACT search query for your next action.

**CRITICAL: Use think_tool before calling ResearchEventsTool to plan your approach, and after each ResearchEventsTool to assess progress. Do not call think_tool two times in a row.**
</Available Tools>

Focus on gathering:
1. **Demographics**: Age, nationality, location, ethnicity, educational background
2. **Professional**: Career history, current role, skills, expertise, accomplishments
3. **Relationships**: Family, spouse, children, colleagues, mentors, business partners
4. **Public Presence**: Social media activity, publications, interviews, media coverage, online presence
5. **Achievements**: Awards, recognition, notable contributions, impact on field/society
6. **Controversies**: Any controversies, legal issues, criticisms, challenges they've faced

**Search Query Strategy:**
- Be specific: Use full name + specific aspect (e.g., "John Doe career history", "Jane Smith awards")
- For public figures: Search for recent news, interviews, profiles
- For professionals: Search for LinkedIn, company pages, industry publications
- For academics: Search for publications, university pages, research profiles
- For entrepreneurs: Search for company information, startup databases, tech news

**CRITICAL:** Execute ONLY ONE tool call now, following the `<Core Execution Cycle>`.
"""

    def get_event_summarizer_prompt(self) -> str:
        return """
Analyze the following people research and identify the 2-3 biggest gaps. Be brief and specific.

**Person Information:**
{existing_data}

<Example Gaps>
- Missing professional background and career history
- Missing personal relationships and family information
- Missing achievements and recognition
- Missing public presence and social media activity
- Missing demographic information
</Example Gaps>

If the research appears complete across all categories, respond with: "Research is COMPLETE."

**Gaps:**
"""

    def get_structure_prompt(self) -> str:
        return """You are a biographical research specialist. Convert the people research data into structured JSON.

<Task>
Extract key facts from the research data and structure them as JSON. Each fact should have:
- category: Which aspect of the person (demographics, professional, relationships, public_presence, achievements, controversies)
- title: A short descriptive title
- content: The detailed information
- source_date: When this information was published (if available, otherwise null)

Also provide:
- person_name: The person's full name
- summary: A 2-3 sentence overview of who this person is and why they're notable
</Task>

<People Research Data>
----
{existing_data}
----
</People Research Data>

CRITICAL: Return only the structured JSON output. No commentary or explanations.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        return PeopleProfile

    def get_initial_data_structure(self) -> PeopleData:
        return PeopleData(
            demographics="",
            professional="",
            relationships="",
            public_presence="",
            achievements="",
            controversies="",
        )

    async def structure_output(
        self, existing_data: PeopleData, config: RunnableConfig
    ) -> dict:
        """Structures people research into a profile."""
        print("--- Structuring People Profile ---")

        if not existing_data:
            print("Warning: No people data found in state")
            return {"structured_output": None}

        # Combine all people data
        combined_data = f"""
Demographics:
{existing_data.demographics}

Professional:
{existing_data.professional}

Relationships:
{existing_data.relationships}

Public Presence:
{existing_data.public_presence}

Achievements:
{existing_data.achievements}

Controversies:
{existing_data.controversies}
"""

        structured_llm = create_llm_structured_model(
            config=config, class_name=PeopleProfile
        )

        prompt = self.get_structure_prompt().format(existing_data=combined_data)
        response = await structured_llm.ainvoke(prompt)

        return {"structured_output": response}
