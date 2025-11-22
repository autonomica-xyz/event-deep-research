import asyncio
from typing import Literal

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from src.configuration import Configuration
from src.llm_service import (
    create_llm_structured_model,
    create_llm_with_tools,
)
from src.research_events.research_events_graph import research_events_app
from src.research_types import ResearchTypeRegistry
from src.state import (
    FinishResearchTool,
    ResearchEventsTool,
    SupervisorState,
    SupervisorStateInput,
)
from src.utils import get_buffer_string_with_tools, get_langfuse_handler, think_tool

config = Configuration()
MAX_TOOL_CALL_ITERATIONS = config.max_tool_iterations


# Verify connection
# if langfuse.auth_check():
#     print("Langfuse client is authenticated and ready!")
# else:
#     print("Authentication failed. Please check your credentials and host.")


async def supervisor_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor_tools"]]:
    """The 'brain' of the agent. It decides the next action.

    This node is now generic and works with any research type by loading
    the appropriate research type and using its prompts.
    """
    # Get research type from state
    research_type_name = state.get("research_type", "biography")
    research_type = ResearchTypeRegistry.get(research_type_name)

    # Standard tools available to all research types
    tools = [
        ResearchEventsTool,
        FinishResearchTool,
        think_tool,
    ]

    # Add any research-type-specific tools
    tools.extend(research_type.get_additional_tools())

    tools_model = create_llm_with_tools(tools=tools, config=config)
    messages = state.get("conversation_history", "")
    messages_summary = get_buffer_string_with_tools(messages)
    last_message = ""
    if len(messages_summary) > 0:
        last_message = messages[-1]

    # Use research type's supervisor prompt
    supervisor_prompt = research_type.get_supervisor_prompt()
    system_message = SystemMessage(
        content=supervisor_prompt.format(
            research_subject=state["research_subject"],
            data_summary=state.get("data_summary", "Everything is missing"),
            last_message=last_message,
        )
    )

    human_message = HumanMessage(content="Start the research process.")
    prompt = [system_message, human_message]

    response = await tools_model.ainvoke(prompt)

    # The output is an AIMessage with tool_calls, which we add to the history
    return Command(
        goto="supervisor_tools",
        update={
            "conversation_history": [response],
            "iteration_count": state.get("iteration_count", 0) + 1,
        },
    )


async def supervisor_tools_node(
    state: SupervisorState,
    config: RunnableConfig,
) -> Command[Literal["supervisor", "structure_output"]]:
    """The 'hands' of the agent. Executes tools and returns a Command for routing.

    This node now supports parallel execution of ResearchEventsTool calls,
    enabling breadth-first research across multiple domains simultaneously.
    """
    # Get research type from state
    research_type_name = state.get("research_type", "biography")
    research_type = ResearchTypeRegistry.get(research_type_name)

    # Get existing data (generic, could be any structure)
    existing_data = state.get("existing_data")
    if existing_data is None:
        existing_data = research_type.get_initial_data_structure()

    data_summary = state.get("data_summary", "")
    used_domains = state.get("used_domains", [])
    last_message = state["conversation_history"][-1]
    iteration_count = state.get("iteration_count", 0)
    exceeded_allowed_iterations = iteration_count >= MAX_TOOL_CALL_ITERATIONS

    # If the LLM made no tool calls, we finish.
    if not last_message.tool_calls or exceeded_allowed_iterations:
        return Command(goto="structure_output")

    # Separate tool calls by type for parallel execution
    all_tool_messages = []
    research_tool_calls = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "FinishResearchTool":
            return Command(goto="structure_output")

        elif tool_name == "think_tool":
            # The 'think' tool is special: it just records a reflection.
            response_content = tool_args["reflection"]
            all_tool_messages.append(
                ToolMessage(
                    content=response_content,
                    tool_call_id=tool_call["id"],
                    name=tool_name,
                )
            )

        elif tool_name == "ResearchEventsTool":
            # Collect research tool calls for parallel execution
            research_tool_calls.append(tool_call)

    # Execute all research tool calls in parallel for breadth-first research
    if research_tool_calls:
        print(f"--- Executing {len(research_tool_calls)} research queries in parallel ---")

        async def execute_research(tool_call):
            """Execute a single research query."""
            research_question = tool_call["args"]["research_question"]
            print(f"  → Researching: {research_question}")

            result = await research_events_app.ainvoke(
                {
                    "research_question": research_question,
                    "existing_data": existing_data,
                    "used_domains": used_domains,
                }
            )
            return result, tool_call["id"]

        # Execute all research queries in parallel
        research_results = await asyncio.gather(
            *[execute_research(tc) for tc in research_tool_calls]
        )

        # Merge results from parallel research
        for result, tool_call_id in research_results:
            existing_data = result["existing_data"]
            used_domains.extend(result["used_domains"])

            all_tool_messages.append(
                ToolMessage(
                    content="Called ResearchEventsTool and returned research data",
                    tool_call_id=tool_call_id,
                    name="ResearchEventsTool",
                )
            )

        # Deduplicate used_domains
        used_domains = list(set(used_domains))

        # Generate summary after all parallel research is complete
        summarizer_prompt = research_type.get_event_summarizer_prompt()
        summarizer_formatted = summarizer_prompt.format(existing_data=existing_data)
        response = await create_llm_structured_model(config=config).ainvoke(
            summarizer_formatted
        )
        data_summary = response.content

        print(f"--- Parallel research complete. Summary: {data_summary[:100]}... ---")

    # The Command helper tells the graph where to go next and what state to update.
    return Command(
        goto="supervisor",
        update={
            "existing_data": existing_data,
            "conversation_history": all_tool_messages,
            "used_domains": used_domains,
            "data_summary": data_summary,
        },
    )


async def structure_output(
    state: SupervisorState, config: RunnableConfig
) -> Command[Literal["__end__"]]:
    """Structures the accumulated research data into final output format.

    This node is now generic and delegates to the research type's
    structure_output method to handle type-specific structuring logic.

    Args:
        state: Current researcher state with accumulated research data.
        config: Runtime configuration with model settings.

    Returns:
        Dictionary containing the structured output for this research type.
    """
    print("--- Structuring Research Output ---")

    # Get research type from state
    research_type_name = state.get("research_type", "biography")
    research_type = ResearchTypeRegistry.get(research_type_name)

    # Get the accumulated research data
    existing_data = state.get("existing_data")

    if not existing_data:
        print("Warning: No research data found in state")
        return {"structured_output": None}

    # Delegate to the research type's structure_output method
    result = await research_type.structure_output(existing_data, config)

    return result


workflow = StateGraph(SupervisorState, input_schema=SupervisorStateInput)

# Add the three core nodes
workflow.add_node("supervisor", supervisor_node)
workflow.add_node("supervisor_tools", supervisor_tools_node)
workflow.add_node("structure_output", structure_output)  # Renamed from structure_events

workflow.add_edge(START, "supervisor")

graph = workflow.compile().with_config({"callbacks": [get_langfuse_handler()]})
