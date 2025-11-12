"""Market research type implementation.

This module implements research for market analysis and industry trends.
It extracts structured information about:
- Market overview and size
- Key players and competition
- Trends and drivers
- Challenges and opportunities
- Future outlook
"""

from typing import Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.llm_service import create_llm_structured_model
from src.research_types.base import ResearchType


class MarketData(BaseModel):
    """Market research data organized by analysis type."""

    overview: str = Field(
        default="",
        description="Market definition, size, value, growth rate, and general description.",
    )
    players: str = Field(
        default="",
        description="Key companies, market leaders, competitive landscape, and market share.",
    )
    trends: str = Field(
        default="",
        description="Current trends, growth drivers, technological developments, and innovations.",
    )
    challenges: str = Field(
        default="",
        description="Market challenges, barriers to entry, risks, and obstacles.",
    )
    outlook: str = Field(
        default="",
        description="Future predictions, growth forecasts, opportunities, and market direction.",
    )


class MarketInsight(BaseModel):
    """A single market insight or finding."""

    category: str = Field(
        description="Category (overview, players, trends, challenges, outlook)"
    )
    title: str = Field(description="Short descriptive title")
    insight: str = Field(description="The key insight or finding")
    data_points: list[str] = Field(
        default_factory=list, description="Supporting data points or statistics"
    )


class MarketReport(BaseModel):
    """Complete market research report."""

    market_name: str = Field(description="Name of the market or industry")
    insights: list[MarketInsight] = Field(description="List of market insights")


class MarketResearchType(ResearchType):
    """Research type for market analysis and industry research."""

    @property
    def name(self) -> str:
        return "market"

    @property
    def display_name(self) -> str:
        return "Market Research"

    def get_subject_display_name(self) -> str:
        return "market"

    def get_supervisor_prompt(self) -> str:
        return """
You are a meticulous market research analyst. Your primary directive is to build a comprehensive market analysis for: **{research_subject}**.

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
*   `ResearchEventsTool`: Searches for market information and analysis.
*   `FinishResearchTool`: Ends the research process. Call this ONLY when the research is complete.
*   `think_tool`:  Use this to analyze results and plan the EXACT search query for your next action.

**CRITICAL: Use think_tool before calling ResearchEventsTool to plan your approach, and after each ResearchEventsTool to assess progress. Do not call think_tool two times in a row.**
</Available Tools>

Focus on gathering:
1. Market overview (size, value, growth rate)
2. Key players and competitive landscape
3. Market trends and growth drivers
4. Challenges and barriers
5. Future outlook and opportunities

**CRITICAL:** Execute ONLY ONE tool call now, following the `<Core Execution Cycle>`.
"""

    def get_event_summarizer_prompt(self) -> str:
        return """
Analyze the following market research data and identify the 2 biggest gaps. Be brief and specific.

**Market Data:**
{existing_data}

<Example Gaps>
- Missing competitive landscape analysis
- Missing future growth projections
</Example Gaps>

**Gaps:**
"""

    def get_structure_prompt(self) -> str:
        return """You are a market analysis specialist. Convert the market research data into structured insights.

<Task>
Extract key insights from the research data and structure them as JSON. Each insight should have:
- category: Which aspect of the market (overview, players, trends, challenges, outlook)
- title: A short descriptive title
- insight: The key finding or insight
- data_points: Supporting statistics or data points (as a list)
</Task>

<Market Research Data>
----
{existing_data}
----
</Market Research Data>

CRITICAL: Return only the structured JSON output. No commentary or explanations.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        return MarketReport

    def get_initial_data_structure(self) -> MarketData:
        return MarketData(
            overview="", players="", trends="", challenges="", outlook=""
        )

    async def structure_output(
        self, existing_data: MarketData, config: RunnableConfig
    ) -> dict:
        """Structures market research into a report.

        Args:
            existing_data: MarketData containing accumulated research
            config: Runtime configuration

        Returns:
            Dictionary with 'structured_output' key containing MarketReport
        """
        print("--- Structuring Market Research Report ---")

        if not existing_data:
            print("Warning: No market data found in state")
            return {"structured_output": None}

        # Combine all market data
        combined_data = f"""
Market Overview:
{existing_data.overview}

Key Players:
{existing_data.players}

Trends & Drivers:
{existing_data.trends}

Challenges:
{existing_data.challenges}

Future Outlook:
{existing_data.outlook}
"""

        structured_llm = create_llm_structured_model(
            config=config, class_name=MarketReport
        )

        prompt = self.get_structure_prompt().format(existing_data=combined_data)
        response = await structured_llm.ainvoke(prompt)

        return {"structured_output": response}
