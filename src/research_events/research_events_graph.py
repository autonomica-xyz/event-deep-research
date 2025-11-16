from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel, Field
from src.configuration import Configuration
from src.llm_service import create_llm_structured_model
from src.research_events.merge_events.merge_events_graph import merge_events_app
from src.search_providers import SearchProviderRegistry
from src.services.url_service import URLService
from src.url_crawler.url_krawler_graph import url_crawler_app
from src.utils import get_langfuse_handler


class InputResearchEventsState(TypedDict):
    """Input state for research events graph.

    Now generic to support any research type (not just biography).
    """

    research_question: str
    existing_data: Any  # Changed from existing_events - can be any structure
    used_domains: list[str]


class ResearchEventsState(InputResearchEventsState):
    """State for research events graph with intermediate fields."""

    urls: list[str]
    extracted_data: str  # Changed from extracted_events


class OutputResearchEventsState(TypedDict):
    """Output state for research events graph."""

    existing_data: Any  # Changed from existing_events
    used_domains: list[str]


class BestUrls(BaseModel):
    selected_urls: list[str] = Field(description="A list of the two best URLs.")


async def url_finder(
    state: ResearchEventsState,
    config: RunnableConfig,
) -> Command[Literal["should_process_url_router"]]:
    """Find URLs for the research question using configured search provider.

    This function now uses the pluggable search provider system, allowing you to
    use Tavily, Brave, DuckDuckGo, SearXNG, or custom search providers.
    """
    research_question = state.get("research_question", "")
    used_domains = state.get("used_domains", [])

    if not research_question:
        raise ValueError("research_question is required")

    # Get configuration to check for search provider preference
    configuration = Configuration.from_runnable_config(config)
    search_provider_name = configuration.search_provider

    # Get search provider (uses configured provider or auto-detects)
    search_provider = SearchProviderRegistry.get(search_provider_name)
    print(f"Using search provider: {search_provider.name}")

    # Perform search
    search_results = await search_provider.search(
        query=research_question,
        max_results=6,
        excluded_domains=used_domains,
    )

    # Extract URLs from search results
    urls = [result.url for result in search_results]

    if not urls:
        print(f"Warning: No search results found for query: {research_question}")
        return Command(goto=END, update={"urls": []})

    # Use LLM to select the best URLs for this research
    prompt = """
        From the search results below, select the two URLs that will provide the most relevant information
        about the research question. Look for authoritative sources, detailed content, and relevance.

        <Search Results>
        {results}
        </Search Results>

        <Research Question>
        {research_question}
        </Research Question>

    """

    # Format URLs with titles for better LLM selection
    formatted_results = "\n".join([
        f"- {result.url} | {result.title}"
        for result in search_results
    ])

    prompt = prompt.format(results=formatted_results, research_question=research_question)

    structured_llm = create_llm_structured_model(config=config, class_name=BestUrls)
    structured_result = structured_llm.invoke(prompt)

    return Command(
        goto="should_process_url_router",
        update={"urls": structured_result.selected_urls},
    )


def updateUrlList(
    state: ResearchEventsState,
) -> tuple[list[str], list[str]]:
    urls = state.get("urls", [])
    used_domains = state.get("used_domains", [])

    return URLService.update_url_list(urls, used_domains)


def should_process_url_router(
    state: ResearchEventsState,
) -> Command[Literal["crawl_url", "__end__"]]:
    urls = state.get("urls", [])
    used_domains = state.get("used_domains", [])

    if urls and len(urls) > 0:
        domain = URLService.extract_domain(urls[0])
        if domain in used_domains:
            # remove first url
            remaining_urls = urls[1:]
            return Command(
                goto="should_process_url_router",
                update={"urls": remaining_urls, "used_domains": used_domains},
            )

        print(f"URLs remaining: {len(state['urls'])}. Routing to crawl.")
        return Command(goto="crawl_url")
    else:
        print("No URLs remaining. Routing to __end__.")
        # Otherwise, end the graph execution
        return Command(
            goto=END,
        )


async def crawl_url(
    state: ResearchEventsState,
) -> Command[Literal["merge_events_and_update"]]:
    """Crawls the next URL and updates the temporary state with new data."""
    urls = state["urls"]
    url_to_process = urls[0]  # Always process the first one
    research_question = state.get("research_question", "")

    if not research_question:
        raise ValueError("research_question is required for url crawling")

    # Invoke the crawler subgraph
    result = await url_crawler_app.ainvoke(
        {"url": url_to_process, "research_question": research_question}
    )
    extracted_data = result.get("extracted_events", "")  # Crawler still uses old name
    # Go to the merge node, updating the state with the extracted data
    return Command(
        goto="merge_events_and_update",
        update={"extracted_data": extracted_data},
    )


async def merge_events_and_update(
    state: ResearchEventsState,
) -> Command[Literal["should_process_url_router"]]:
    """Merges new data, removes the processed URL, and loops back to the router."""
    existing_data = state.get("existing_data")
    extracted_data = state.get("extracted_data", "")
    research_question = state.get("research_question", "")

    # Invoke the merge subgraph
    result = await merge_events_app.ainvoke(
        {
            "existing_data": existing_data,
            "extracted_data": extracted_data,
            "research_question": research_question,
        }
    )

    remaining_urls, used_domains = updateUrlList(state)

    # Remaining URLs after removal
    return Command(
        goto="should_process_url_router",
        update={
            "existing_data": result["existing_data"],
            "urls": remaining_urls,
            "used_domains": used_domains,
        },
    )


research_events_builder = StateGraph(
    ResearchEventsState,
    input_schema=InputResearchEventsState,
    output_schema=OutputResearchEventsState,
    config_schema=Configuration,
)

# Add all the nodes to the graph
research_events_builder.add_node("url_finder", url_finder)
research_events_builder.add_node("should_process_url_router", should_process_url_router)
research_events_builder.add_node("crawl_url", crawl_url)
research_events_builder.add_node("merge_events_and_update", merge_events_and_update)

# Set the entry point
research_events_builder.add_edge(START, "url_finder")


research_events_app = research_events_builder.compile().with_config(
    {"callbacks": [get_langfuse_handler()]}
)
