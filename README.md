# Code Delivery for Module #5 | Cyclic agent with persistent memory

Async LangGraph agent that answers questions about a competence catalog. The model decides which catalog query to call. Tool results come back into the same node, so one question can take several steps. `AsyncSqliteSaver` keeps that conversation under a `thread_id`. The catalog itself lives in a second SQLite file.

This delivery does not classify uploads and it does not build a training plan. Those flows are out of scope. The catalog is seeded, and the agent only reads it.

## Graph

Diagram: [`diagrams/graph.mermaid`](diagrams/graph.mermaid)

1. The user sends a question. It is appended to the message state for the current `thread_id`.
2. The `llm` node calls the chat model with the system prompt and the messages stored so far. The tools are bound to that model (`bind_tools`). There is no application branch on the text of the question.
3. `tools_condition` reads the model reply. If it contains tool calls, the edge goes to `tools`. If it does not, the edge goes to `END`.
4. `tools` runs the catalog functions from `tools.py` and appends each result as a tool message.
5. The edge from `tools` returns to `llm`. The model sees the observation and either calls another tool or answers.
6. `recursion_limit` (10 by default) stops a run that never settles.

The same `thread_id` reloads the checkpoint, so a follow-up such as "¿Y para Liderazgo?" still refers to the previous question. A different `thread_id` does not.

## Two SQLite files

| File | Role |
|------|------|
| `data/catalog.sqlite` | Core catalog: assets, competences, the asset-competence map, and training courses. The tools read this file. `--fixtures` rebuilds it from `data/fixtures.csv`. |
| `data/checkpoints.sqlite` | LangGraph checkpointer (`AsyncSqliteSaver`). One conversation state per `thread_id`. |

Neither file is committed. `trace.log` is the local execution trace and is gitignored too. The CLI and `python main.py --demo` both append each turn there. `-c` / `--clean` truncates that file before the run.

## Requirements

- Python 3.12
- An API key for the chat provider you call:
  - `OPENAI_API_KEY` when `-l openai` (the default)
  - `ANTHROPIC_API_KEY` when `-l anthropic`
  - `GEMINI_API_KEY` when `-l gemini`

## Setup

From this directory:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.dist .env
```

Put the key for the provider you call in `.env`. That file is gitignored.

## Project layout

| File | Role |
|------|------|
| `main.py` | Loads fixtures, runs the demo, or reads questions until `exit` or `quit` |
| `agent.py` | `StateGraph` on `MessagesState`, model node, tool node, conditional edge, trace formatting |
| `db.py` | SQLAlchemy queries over the core catalog |
| `tools.py` | `@tool` wrappers the agent binds to the model |
| `models.py` | `Asset`, `Competence`, `AssetCompetenceMap`, `TrainingCourse` |
| `fixtures.py` | `load_fixtures`: replaces the core catalog from the CSV |
| `data/fixtures.csv` | Seed rows for the four tables |
| `samples.py` | Demo questions, including a follow-up on the same thread |
| `llm.py` | `OpenAIClient`, `AnthropicClient`, and `GeminiClient` |
| `factory.py` | Builds the chat client from `LLMConfig` |
| `schemas.py` | `LLMProvider`, `LLMConfig`, and the tool input models |
| `settings.py` | Paths, `recursion_limit`, excerpt length, and the default provider |
| `diagrams/graph.mermaid` | The agent graph and the two databases |
| `trace.log` | Execution trace. The CLI and `--demo` append each turn here. Gitignored. `-c` / `--clean` truncates it. |
| `requirements.txt` | Dependencies |
| `.env.dist` | Env template with empty secrets |

## Catalog

`human_check` on `AssetCompetenceMap` is `approved`, `rejected`, or empty. Empty means nobody has reviewed that link yet. The tools call that state `pending`.

`list_assets_by_competence` returns the asset name, type, url, `ai_check`, and `human_check`. It does not return the transcription. `get_asset` returns the url, every review link, and a transcription excerpt (`TRANSCRIPT_EXCERPT_CHARS`, 400 by default). A list followed by a fetch is the multi-step path.

| Tool | When the model should use it |
|------|------------------------------|
| `get_competence` | Resolve a competence name to its id and description. On a miss, the payload lists the valid names. |
| `list_assets_by_competence` | Assets linked to a competence, optionally filtered by human review. No transcription. |
| `get_asset` | One asset: url, checks, transcription excerpt. Several matches come back as names to retry with. |
| `list_trainings_by_user` | Courses assigned to an email. |
| `list_trainings_competence` | Courses activated for a competence. |
| `list_trainings_by_user_and_competence` | Courses for one person and one competence, including progress as an integer percentage from 0 to 100. |

A tool that cannot resolve a name returns JSON with `error` and the valid values. The model is told to retry with one of those values or ask the user. It does not receive a stack trace, and the graph does not catch that case with an `if` on the question.

## Usage

Run from this directory. The first run needs `--fixtures` so the core catalog exists.

```bash
python main.py --fixtures

# Questions from the terminal. exit or quit stops.
# thread_id cli is reused across runs until data/checkpoints.sqlite is deleted.
python main.py

python main.py -l anthropic
python main.py --llm gemini

# Load the catalog and run samples.py on one fresh thread.
python main.py -f --demo

# Truncate trace.log, then run the samples.
python main.py -c --demo
```

`--fixtures` / `-f` rebuilds `data/catalog.sqlite` and then continues into the demo or the prompt. It does not modify `data/checkpoints.sqlite`. `-c` / `--clean` truncates `trace.log` before that run. Without it, new turns are appended.

### Demo

`samples.py` asks six questions on one `thread_id`:

1. Empatía videos a human already approved, and the transcription of the first one. The list tool has no transcription, so the model has to call `get_asset` for the first name.
2. "¿Y para Liderazgo?" The checkpoint still holds the previous question, and the model has to query Liderazgo rather than reuse the Empatía rows.
3. A competence that is not in the catalog (`Resiliencia`). The tool returns the valid names.
4. `El silencio en la mesa`, whose transcription is empty and whose AI check is false.
5. Every training assigned to `ana@nimbus.test`.
6. "¿Y cuál es su progreso en Empatía?" The email is not repeated. The checkpoint still has Ana's list from the previous turn.

The people in the fixtures are Ana Pérez (`ana@nimbus.test`), Luis Gómez (`luis@nimbus.test`), and María Ruiz (`maria@nimbus.test`).

## Trace

Each turn is appended to `trace.log` (gitignored), from the CLI and from `--demo`. A turn records the time, the `thread_id`, how many tool results came back, the user text, each tool call with its arguments, each tool payload, and the final answer. `-c` / `--clean` truncates that file first.

To drop the remembered CLI session, delete `data/checkpoints.sqlite`. To rebuild the catalog, run `python main.py --fixtures`.
