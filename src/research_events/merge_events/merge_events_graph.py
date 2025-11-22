from typing import Any, List, Literal, TypedDict

from langchain_core.tools import tool
from langgraph.graph import START, StateGraph
from langgraph.graph.state import Command, RunnableConfig
from langgraph.pregel.main import asyncio
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_with_tools
from src.research_events.chunk_graph import create_biographic_event_graph
from src.research_events.merge_events.prompts import (
    EXTRACT_AND_CATEGORIZE_PROMPT,
    MERGE_EVENTS_TEMPLATE,
)
from src.research_events.merge_events.utils import ensure_categories_with_events
from src.research_events.merge_events.utils import (
    ensure_categories_with_events as ensure_data_structure,
)
from src.services.event_service import EventService
from src.state import CategoriesWithEvents
from src.url_crawler.utils import chunk_text_by_tokens
from src.utils import get_langfuse_handler


class RelevantEventsCategorized(BaseModel):
    """The chunk contains relevant biographical events that have been categorized."""

    early: str = Field(
        description="Bullet points of events related to childhood, upbringing, family, education, and early influences"
    )
    personal: str = Field(
        description="Bullet points of events related to relationships, friendships, family life, residence, and personal traits"
    )
    career: str = Field(
        description="Bullet points of events related to professional journey, publications, collaborations, and milestones"
    )
    legacy: str = Field(
        description="Bullet points of events related to recognition, impact, influence, and how they are remembered"
    )


class IrrelevantChunk(BaseModel):
    """The chunk contains NO biographical events relevant to the research question."""


class InputMergeEventsState(TypedDict):
    """The complete state for the enhanced event merging sub-graph.

    Now uses generic field names to support multiple research types.
    """

    existing_data: Any  # Changed from existing_events - can be any structure
    extracted_data: str  # Changed from extracted_events
    research_question: str


class MergeEventsState(InputMergeEventsState):
    text_chunks: List[str]  # token-based chunks
    categorized_chunks: List[Any]  # results per chunk (structure depends on research type)
    extracted_data_categorized: Any  # Changed from extracted_events_categorized


class OutputMergeEventsState(TypedDict):
    existing_data: Any  # includes the existing data + the data from the new extraction


async def split_events(
    state: MergeEventsState,
) -> Command[Literal["filter_chunks", "__end__"]]:
    """Use token-based chunking from URL crawler and filter for relevant data"""
    extracted_data = state.get("extracted_data", "")

    if not extracted_data.strip():
        # No content to process
        return Command(
            goto="__end__",
            update={"text_chunks": [], "categorized_chunks": []},
        )

    chunks = await chunk_text_by_tokens(extracted_data)

    return Command(
        goto="filter_chunks",
        update={"text_chunks": chunks[0:20], "categorized_chunks": []},
    )


async def filter_chunks(
    state: MergeEventsState, config: RunnableConfig
) -> Command[Literal["extract_and_categorize_chunk", "__end__"]]:
    """Filter chunks to only process those containing biographical events"""
    chunks = state.get("text_chunks", [])

    if not chunks:
        return Command(
            goto="__end__",
        )

    # Use chunk graph to filter for biographical events
    chunk_graph = create_biographic_event_graph()

    configurable = Configuration.from_runnable_config(config)
    if len(chunks) > configurable.max_chunks:
        # To avoid recursion issues, set max chunks
        chunks = chunks[: configurable.max_chunks]

    # Process each chunk through the biographic event detection graph
    relevant_chunks = []
    for chunk in chunks:
        chunk_result = await chunk_graph.ainvoke({"text": chunk}, config)

        # Check if any chunk contains biographical events
        has_events = any(
            result.contains_biographic_event
            for result in chunk_result["results"].values()
        )
        print(f"contains_biographic_event: {has_events}")

        if has_events:
            relevant_chunks.append(chunk)

    if not relevant_chunks:
        # No relevant chunks found
        return Command(goto="__end__")

    return Command(
        goto="extract_and_categorize_chunk",
        update={"text_chunks": chunks, "categorized_chunks": []},
    )


async def extract_and_categorize_chunk(
    state: MergeEventsState, config: RunnableConfig
) -> Command[Literal["extract_and_categorize_chunk", "merge_categorizations"]]:
    """Combined extraction and categorization"""
    chunks = state.get("text_chunks", [])
    categorized_chunks = state.get("categorized_chunks", [])

    if len(categorized_chunks) >= len(chunks):
        # all categorized_chunks done → move to merge
        return Command(goto="merge_categorizations")

    # take next chunk
    chunk = chunks[len(categorized_chunks)]
    research_question = state.get("research_question", "")

    prompt = EXTRACT_AND_CATEGORIZE_PROMPT.format(
        # research_question=research_question,
        text_chunk=chunk
    )

    tools = [tool(RelevantEventsCategorized), tool(IrrelevantChunk)]
    model = create_llm_with_tools(tools=tools, config=config)
    response = await model.ainvoke(prompt)

    # Parse response
    if (
        response.tool_calls
        and response.tool_calls[0]["name"] == "RelevantEventsCategorized"
    ):
        categorized_data = response.tool_calls[0]["args"]
        # Convert any list values to strings
        categorized_data = {
            k: "\n".join(v) if isinstance(v, list) else v
            for k, v in categorized_data.items()
        }
        categorized = CategoriesWithEvents(**categorized_data)
    else:
        categorized = CategoriesWithEvents(early="", personal="", career="", legacy="")

    return Command(
        goto="extract_and_categorize_chunk",  # loop until all chunks processed
        update={"categorized_chunks": categorized_chunks + [categorized]},
    )


async def merge_categorizations(
    state: MergeEventsState,
) -> Command[Literal["combine_new_and_original_data"]]:
    """Merge all categorized chunks into a single data structure"""
    results = state.get("categorized_chunks", [])

    merged = EventService.merge_categorized_events(results)

    return Command(
        goto="combine_new_and_original_data",
        update={"extracted_data_categorized": merged},
    )


async def combine_new_and_original_data(
    state: MergeEventsState, config: RunnableConfig
) -> Command:
    """Merge original and new data for each field using an LLM.

    This function is now GENERIC and works with any Pydantic model.
    It dynamically detects the data type and merges all string fields.
    """
    print("Combining new and original data...")

    existing_data_raw = state.get("existing_data")
    new_data_raw = state.get("extracted_data_categorized")

    # If no existing data, use new data
    if not existing_data_raw:
        if not new_data_raw:
            print("No data to merge. Returning empty.")
            return Command(goto="__end__", update={"existing_data": None})
        print("No existing data. Using new data.")
        return Command(goto="__end__", update={"existing_data": new_data_raw})

    # Determine data type - could be CategoriesWithEvents, PeopleData, etc.
    data_type = type(existing_data_raw)

    # For biography compatibility: ensure it's CategoriesWithEvents if needed
    if data_type.__name__ == "CategoriesWithEvents":
        existing_data = ensure_categories_with_events(existing_data_raw)
        new_data = ensure_categories_with_events(new_data_raw) if new_data_raw else existing_data
    else:
        # For other types (PeopleData, CompanyData, etc.), use as-is
        existing_data = existing_data_raw
        new_data = new_data_raw if new_data_raw else existing_data

    # Check if there's any new data to merge
    if not new_data:
        print("No new data found. Keeping existing data.")
        return Command(goto="__end__", update={"existing_data": existing_data})

    # Get all field names from the model dynamically
    field_names = data_type.model_fields.keys()

    # Check if there's any non-empty new data
    has_new_data = any(
        getattr(new_data, field, "").strip() for field in field_names
    )

    if not has_new_data:
        print("No new data found. Keeping existing data.")
        return Command(goto="__end__", update={"existing_data": existing_data})

    # Merge each field in parallel
    merge_tasks = []

    for field_name in field_names:
        existing_text = getattr(existing_data, field_name, "").strip()
        new_text = getattr(new_data, field_name, "").strip()

        if not (existing_text or new_text):
            continue  # nothing to merge in this field

        existing_display = existing_text if existing_text else "No data"
        new_display = new_text if new_text else "No data"

        prompt = MERGE_EVENTS_TEMPLATE.format(
            original=existing_display, new=new_display
        )

        # Use regular structured model for merging (not tools model)
        from src.llm_service import create_llm_structured_model

        merge_tasks.append(
            (field_name, create_llm_structured_model(config=config).ainvoke(prompt))
        )

    final_merged_dict = {}
    if merge_tasks:
        field_names_to_merge, tasks = zip(*merge_tasks)
        responses = await asyncio.gather(*tasks)
        final_merged_dict = {
            field: resp.content for field, resp in zip(field_names_to_merge, responses)
        }

    # Ensure all fields are included (even ones that weren't merged)
    for field_name in field_names:
        if field_name not in final_merged_dict:
            final_merged_dict[field_name] = getattr(existing_data, field_name, "")

    # Create new instance of the same type with merged data
    final_merged_output = data_type(**final_merged_dict)
    print(f"--- Merged data into {data_type.__name__} ---")
    return Command(goto="__end__", update={"existing_data": final_merged_output})


merge_events_graph_builder = StateGraph(
    MergeEventsState, input_schema=InputMergeEventsState, config_schema=Configuration
)

merge_events_graph_builder.add_node("split_events", split_events)
merge_events_graph_builder.add_node("filter_chunks", filter_chunks)
merge_events_graph_builder.add_node(
    "extract_and_categorize_chunk", extract_and_categorize_chunk
)
merge_events_graph_builder.add_node("merge_categorizations", merge_categorizations)
merge_events_graph_builder.add_node(
    "combine_new_and_original_data", combine_new_and_original_data  # Renamed
)

merge_events_graph_builder.add_edge(START, "split_events")


merge_events_app = merge_events_graph_builder.compile().with_config(
    {
        "callbacks": [get_langfuse_handler()],
        "recursionLimit": 200,
    },
)
