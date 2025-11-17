"""Company research type implementation.

This module implements research for companies and organizations.
It extracts structured information about:
- Overview (founding, mission, headquarters)
- Leadership (CEO, executives, board)
- Products & Services
- Financial information
- News & Recent developments
"""

from typing import Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.llm_service import create_llm_structured_model
from src.research_types.base import ResearchType


class CompanyData(BaseModel):
    """Company research data organized by information type."""

    overview: str = Field(
        default="",
        description="Company founding date, mission, vision, headquarters location, and general description.",
    )
    leadership: str = Field(
        default="",
        description="CEO, executives, board members, and key leadership information.",
    )
    products: str = Field(
        default="",
        description="Main products, services, and offerings.",
    )
    financials: str = Field(
        default="",
        description="Revenue, funding, valuation, stock performance, and financial metrics.",
    )
    news: str = Field(
        default="",
        description="Recent news, developments, partnerships, and announcements.",
    )


class CompanyFact(BaseModel):
    """A single fact about the company."""

    category: str = Field(
        description="Category of this fact (overview, leadership, products, financials, news)"
    )
    title: str = Field(description="Short title for this fact")
    content: str = Field(description="Detailed information")
    source_date: str | None = Field(
        None, description="When this information was published or last updated"
    )


class CompanyProfile(BaseModel):
    """Complete company research profile."""

    company_name: str = Field(description="Official company name")
    facts: list[CompanyFact] = Field(description="List of researched facts")


class CompanyResearchType(ResearchType):
    """Research type for companies and organizations."""

    @property
    def name(self) -> str:
        return "company"

    @property
    def display_name(self) -> str:
        return "Company Research"

    def get_subject_display_name(self) -> str:
        return "company"

    def get_supervisor_prompt(self) -> str:
        return """
You are a meticulous business research agent. Your primary directive is to build a comprehensive company profile for: **{research_subject}**.

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
*   `ResearchEventsTool`: Searches for information about the company.
*   `FinishResearchTool`: Ends the research process. Call this ONLY when the research is complete.
*   `think_tool`:  Use this to analyze results and plan the EXACT search query for your next action.

**CRITICAL: Use think_tool before calling ResearchEventsTool to plan your approach, and after each ResearchEventsTool to assess progress. Do not call think_tool two times in a row.**
</Available Tools>

Focus on gathering:
1. Company overview (founding, mission, headquarters)
2. Leadership team (CEO, executives)
3. Products and services
4. Financial information (revenue, funding, valuation)
5. Recent news and developments

**CRITICAL:** Execute ONLY ONE tool call now, following the `<Core Execution Cycle>`.
"""

    def get_event_summarizer_prompt(self) -> str:
        return """
Analyze the following company information and identify the 2 biggest gaps. Be brief and specific.

**Company Information:**
{existing_data}

<Example Gaps>
- Missing financial information (revenue, funding)
- Missing recent news and developments
</Example Gaps>

**Gaps:**
"""

    def get_structure_prompt(self) -> str:
        return """You are a data processing specialist. Convert the company research data into structured JSON.

<Task>
Extract key facts from the research data and structure them as JSON. Each fact should have:
- category: Which aspect of the company (overview, leadership, products, financials, news)
- title: A short descriptive title
- content: The detailed information
- source_date: When this information was published (if available)
</Task>

<Company Research Data>
----
{existing_data}
----
</Company Research Data>

CRITICAL: Return only the structured JSON output. No commentary or explanations.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        return CompanyProfile

    def get_initial_data_structure(self) -> CompanyData:
        return CompanyData(
            overview="", leadership="", products="", financials="", news=""
        )

    async def structure_output(
        self, existing_data: CompanyData, config: RunnableConfig
    ) -> dict:
        """Structures company research into a profile.

        Args:
            existing_data: CompanyData containing accumulated research
            config: Runtime configuration

        Returns:
            Dictionary with 'structured_output' key containing CompanyProfile
        """
        print("--- Structuring Company Profile ---")

        if not existing_data:
            print("Warning: No company data found in state")
            return {"structured_output": None}

        # Combine all company data
        combined_data = f"""
Overview:
{existing_data.overview}

Leadership:
{existing_data.leadership}

Products:
{existing_data.products}

Financials:
{existing_data.financials}

News:
{existing_data.news}
"""

        structured_llm = create_llm_structured_model(
            config=config, class_name=CompanyProfile
        )

        prompt = self.get_structure_prompt().format(existing_data=combined_data)
        response = await structured_llm.ainvoke(prompt)

        return {"structured_output": response}
