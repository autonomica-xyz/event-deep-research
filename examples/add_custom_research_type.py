"""Example: How to add a custom research type.

This example shows you how to create and register your own research type
for the deep research system.
"""

from typing import Any, Type

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from src.llm_service import create_llm_structured_model
from src.research_types.base import ResearchType
from src.research_types.registry import ResearchTypeRegistry


# Step 1: Define your output schema
class BookData(BaseModel):
    """Structured data about a book."""

    plot: str = Field(default="", description="Plot summary")
    characters: str = Field(default="", description="Main characters")
    themes: str = Field(default="", description="Major themes")
    reception: str = Field(default="", description="Critical reception and awards")


class BookReview(BaseModel):
    """A structured book review."""

    title: str = Field(description="Book title")
    author: str = Field(description="Book author")
    summary: str = Field(description="Overall summary")
    key_points: list[str] = Field(description="Key points about the book")


# Step 2: Implement your ResearchType
class BookResearchType(ResearchType):
    """Research type for books and literature."""

    @property
    def name(self) -> str:
        return "book"

    @property
    def display_name(self) -> str:
        return "Book Research"

    def get_subject_display_name(self) -> str:
        return "book"

    def get_supervisor_prompt(self) -> str:
        return """
You are a literary research agent. Your primary directive is to build a comprehensive analysis of: **{research_subject}**.

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
*   `ResearchEventsTool`: Searches for information about the book.
*   `FinishResearchTool`: Ends the research process. Call this ONLY when the research is complete.
*   `think_tool`:  Use this to analyze results and plan the EXACT search query for your next action.
</Available Tools>

Focus on gathering:
1. Plot summary
2. Main characters
3. Major themes
4. Critical reception and awards

**CRITICAL:** Execute ONLY ONE tool call now, following the `<Core Execution Cycle>`.
"""

    def get_event_summarizer_prompt(self) -> str:
        return """
Analyze the following book information and identify the 2 biggest gaps. Be brief and specific.

**Book Information:**
{existing_data}

<Example Gaps>
- Missing plot summary
- Missing critical reception information
</Example Gaps>

**Gaps:**
"""

    def get_structure_prompt(self) -> str:
        return """You are a literary analysis specialist. Convert the book research into a structured review.

<Task>
Extract key information and structure it as JSON with:
- title: The book title
- author: The book author
- summary: An overall summary
- key_points: List of key points about the book
</Task>

<Book Research Data>
----
{existing_data}
----
</Book Research Data>

CRITICAL: Return only the structured JSON output. No commentary or explanations.
"""

    def get_output_schema(self) -> Type[BaseModel]:
        return BookReview

    def get_initial_data_structure(self) -> BookData:
        return BookData(plot="", characters="", themes="", reception="")

    async def structure_output(
        self, existing_data: BookData, config: RunnableConfig
    ) -> dict:
        """Structures book research into a review."""
        print("--- Structuring Book Review ---")

        if not existing_data:
            print("Warning: No book data found in state")
            return {"structured_output": None}

        # Combine all book data
        combined_data = f"""
Plot:
{existing_data.plot}

Characters:
{existing_data.characters}

Themes:
{existing_data.themes}

Reception:
{existing_data.reception}
"""

        structured_llm = create_llm_structured_model(
            config=config, class_name=BookReview
        )

        prompt = self.get_structure_prompt().format(existing_data=combined_data)
        response = await structured_llm.ainvoke(prompt)

        return {"structured_output": response}


# Step 3: Register your research type
def register_custom_types():
    """Register all custom research types."""
    ResearchTypeRegistry.register(BookResearchType())
    print("Registered custom research type: book")


# Step 4: Use your custom research type
if __name__ == "__main__":
    import asyncio

    from src.graph import graph

    # Register the custom type
    register_custom_types()

    async def research_book():
        """Research a book."""
        input_data = {
            "research_subject": "To Kill a Mockingbird by Harper Lee",
            "research_type": "book",  # Use your custom type!
        }

        print(f"Starting book research on: {input_data['research_subject']}")
        print("=" * 60)

        result = await graph.ainvoke(input_data)
        structured_output = result.get("structured_output")

        print("\n" + "=" * 60)
        print("Research Complete!")
        print("=" * 60)

        if structured_output:
            print(f"\nTitle: {structured_output.title}")
            print(f"Author: {structured_output.author}")
            print(f"\nSummary: {structured_output.summary}")
            print(f"\nKey Points:")
            for point in structured_output.key_points:
                print(f"  • {point}")

    asyncio.run(research_book())
