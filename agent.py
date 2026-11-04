"""ReAct agent over the competence catalog.

The model node decides when to call a tool. ``tools_condition`` sends the
graph back through the tool node until the model answers without a tool call.
``AsyncSqliteSaver`` stores that state per ``thread_id``. The core catalog
lives in a different SQLite file and is only reached through the tools.
"""

import json
import logging
from datetime import datetime, timezone

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from tools import tools
from factory import LLMFactory
from schemas import LLMConfig
from settings import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are the assistant for a competence-catalog platform.
Reply in the user's language.
Every fact about competences, assets, or trainings must come from a tool. Do not invent names, urls, review states, transcriptions, or progress values.
list_assets_by_competence does not include the transcription. When the user wants to know what an asset says, call get_asset with the exact name of the first item in that list, or of the item they asked for.
When a tool returns an error or incomplete data, call it again with a valid name from that response, or ask the user to clarify. Do not fill the gap from memory.
When they ask for a person's trainings, call list_trainings_by_user. That list already includes each course's competence and progress. Progress is an integer from 0 to 100 and is already a percentage: do not scale it. list_trainings_by_user_and_competence does not replace the full list.
On a follow-up, use the conversation to understand the subject, then query the tools again for the new case.
"""


class AgentState(MessagesState):
    """Conversation state. Messages accumulate with the add_messages reducer."""


def compile_agent(config: LLMConfig, checkpointer: BaseCheckpointSaver) -> CompiledStateGraph:
    """Compile the StateGraph with the model, the catalog tools, and a checkpointer."""
    model = LLMFactory.create_client(config).get_llm_model().bind_tools(tools)

    async def call_model(state: AgentState) -> dict[str, list[BaseMessage]]:
        response = await model.ainvoke(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        )
        return {"messages": [response]}

    workflow = StateGraph(AgentState)
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "llm")
    # tools_condition routes to "tools" when the model requested a call, otherwise END.
    workflow.add_conditional_edges("llm", tools_condition)
    workflow.add_edge("tools", "llm")
    return workflow.compile(checkpointer=checkpointer)


def invoke_config(thread_id: str) -> dict[str, object]:
    """Config for one turn. ``recursion_limit`` caps the ReAct cycle."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.recursion_limit,
    }


async def ask(graph: CompiledStateGraph, question: str, thread_id: str) -> str:
    """Run one user turn and append the formatted trace to ``trace.log``."""
    config = invoke_config(thread_id)
    previous = await _message_count(graph, config)
    try:
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content=question)]},
            config,
        )
    except Exception as exc:
        logger.exception("Agent invocation failed")
        partial = await _partial_turn(graph, config, thread_id, previous)
        failure = format_failure(thread_id, question, exc)
        text = f"{partial}\n{failure}" if partial else failure
        append_trace(text)
        return text

    messages = list(result["messages"][previous:])
    text = format_turn(thread_id, messages)
    append_trace(text)
    return text


def append_trace(text: str) -> None:
    """Append one formatted turn to ``trace.log``."""
    path = settings.trace_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")
        handle.write("\n")


def format_turn(thread_id: str, messages: list[BaseMessage]) -> str:
    """Render one ReAct turn: user text, tool calls, tool results, and the answer."""
    tool_results = sum(isinstance(message, ToolMessage) for message in messages)
    lines = [
        "=" * 80,
        f"{_now()}  thread={thread_id}  tool_results={tool_results}",
        "-" * 80,
    ]
    for message in messages:
        lines.append(_format_message(message))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_failure(thread_id: str, question: str, exc: BaseException) -> str:
    """Render a turn that stopped before a final answer."""
    lines = [
        "=" * 80,
        f"{_now()}  thread={thread_id}  status=error",
        "-" * 80,
        "USER",
        question,
        "",
        f"ERROR {type(exc).__name__}",
        str(exc),
        "",
    ]
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _message_count(graph: CompiledStateGraph, config: dict[str, object]) -> int:
    snapshot = await graph.aget_state(config)
    messages = snapshot.values.get("messages", []) if snapshot.values else []
    return len(messages)


async def _partial_turn(
    graph: CompiledStateGraph,
    config: dict[str, object],
    thread_id: str,
    previous: int,
) -> str:
    try:
        snapshot = await graph.aget_state(config)
    except Exception:
        logger.exception("Could not read the checkpoint after a failed turn")
        return ""
    stored = snapshot.values.get("messages", []) if snapshot.values else []
    if len(stored) <= previous:
        return ""
    return format_turn(thread_id, list(stored[previous:]))


def _format_message(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        return f"USER\n{_text(message)}"
    if isinstance(message, ToolMessage):
        name = message.name or "tool"
        return f"TOOL {name}\n{_text(message)}"
    if isinstance(message, AIMessage):
        return _format_assistant(message)
    return f"{message.type.upper()}\n{_text(message)}"


def _format_assistant(message: AIMessage) -> str:
    lines = ["ASSISTANT"]
    calls = list(message.tool_calls or [])
    if calls:
        lines.append("tool_calls:")
        for call in calls:
            name = str(call.get("name", ""))
            args = call.get("args", {})
            lines.append(f"- {name}")
            lines.append(json.dumps(args, ensure_ascii=False, indent=2, default=str))
    text = _text(message)
    if text:
        if calls:
            lines.append("content:")
        lines.append(text)
    invalid = list(message.invalid_tool_calls or [])
    if invalid:
        lines.append("invalid_tool_calls:")
        lines.append(json.dumps(invalid, ensure_ascii=False, indent=2, default=str))
    if not calls and not text and not invalid:
        lines.append("(empty)")
    return "\n".join(lines)


def _text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                value = block.get("text", "")
                if value:
                    parts.append(str(value))
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return str(content).strip()
