# Load env before the modules that read API keys from the environment.
from dotenv import load_dotenv
load_dotenv()

import argparse
import asyncio
import logging
from datetime import datetime, timezone

from langgraph.graph.state import CompiledStateGraph

from agent import compile_agent, ask
from checkpointer import open_checkpointer
from fixtures import load_fixtures
from samples import DEMO_QUESTIONS
from schemas import LLMConfig, LLMProvider
from settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

CLI_THREAD_ID = "cli"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ask the competence catalog agent. The default mode reads questions until exit or quit."
    )
    parser.add_argument(
        "-l",
        "--llm",
        default=settings.default_provider,
        choices=[provider.value for provider in LLMProvider],
        help="LLM provider (default: openai, or LLM_PROVIDER)",
    )
    parser.add_argument(
        "-f",
        "--fixtures",
        action="store_true",
        help="Load data/fixtures.csv into the core SQLite catalog before starting.",
    )
    parser.add_argument(
        "-d",
        "--demo",
        action="store_true",
        help="Run the sample questions on one fresh thread, write the trace, and exit.",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="Truncate trace.log before running.",
    )
    return parser.parse_args()


def _read_question() -> str:
    try:
        return input("Question: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "quit"


async def _run_demo(graph: CompiledStateGraph) -> None:
    thread_id = "demo-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(
        f"thread_id={thread_id}  recursion_limit={settings.recursion_limit}",
        flush=True,
    )
    for label, question in DEMO_QUESTIONS:
        print(f"\n--- {label} ---", flush=True)
        print(f"Question: {question}", flush=True)
        turn = await ask(graph, question, thread_id)
        print(turn, flush=True)
    logger.info("Trace appended to %s", settings.trace_log_path)
    logger.info("Finished")


async def _run_loop(graph: CompiledStateGraph) -> None:
    print(
        f"thread_id={CLI_THREAD_ID}  recursion_limit={settings.recursion_limit}",
        flush=True,
    )
    print(
        "This session reuses thread_id cli defined in constant `CLI_THREAD_ID`. "
        f"Delete {settings.checkpoint_db_path} to forget the conversation.",
        flush=True,
    )
    print("Write a question and press Enter. exit or quit to terminate.\n", flush=True)
    while True:
        question = _read_question()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        try:
            turn = await ask(graph, question, CLI_THREAD_ID)
        except Exception as exc:
            logger.error("The agent failed: %s", exc)
            continue
        print(turn, flush=True)
    logger.info("Finished")


async def main(args: argparse.Namespace) -> None:
    if args.clean:
        settings.trace_log_path.write_text("", encoding="utf-8")
        logger.info("Trace log reset at %s", settings.trace_log_path)

    if args.fixtures:
        counts = load_fixtures()
        summary = ", ".join(f"{name}={count}" for name, count in counts.items())
        print(f"Loaded fixtures into {settings.core_db_path}: {summary}", flush=True)

    if not settings.core_db_path.is_file():
        logger.error(
            "Core database not found at %s. Run python main.py --fixtures.",
            settings.core_db_path,
        )
        return

    config = LLMConfig(provider=LLMProvider(args.llm))
    logger.info("LLMConfig initialized (provider=%s)", config.provider.value)
    logger.info("Checkpointer backend=%s", settings.checkpoint_backend)

    async with open_checkpointer() as checkpointer:
        graph = compile_agent(config, checkpointer)
        if args.demo:
            await _run_demo(graph)
            return
        await _run_loop(graph)


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
