# Mini multi-agent (proof of concept)

Pipeline: **planner** -> **executor** (draft) -> **critic** (optional) -> **final answer** that uses all prior text. If the critic reports issues, the final step is supposed to fix them; if the critic says OK, the final answer still polishes from the draft.

Same model can back every step; orchestration is explicit string handoff, not a swarm.

Code lives in the `mini_agents/` package; install from this repository root.

## Install (console command `mini-agents`)

From the repo root:

```bash
pip install -e .
```

Ensure `.env` exists in `mini_agents/` (copy `mini_agents/.env.example`) or set env vars. Then:

```bash
mini-agents
```

Opens an interactive prompt; type questions or slash commands (`/help`). One-shot:

```bash
mini-agents "Your question here"
```

Without install:

```bash
cd mini_agents
pip install -r requirements.txt
python main.py
python main.py "Your question"
```

From repo root without editable install:

```bash
python -m mini_agents
```

## Ollama

Install [Ollama](https://ollama.com), `ollama pull qwen2.5:7b`, copy `mini_agents/.env.example` to `mini_agents/.env`.

CPU-only (reliable): stop Ollama, then start with `OLLAMA_NUM_GPU=0` (see `/help` in the app). Per-request `cpu` may work on newer Ollama via OpenAI `options`; if not, use the server env.

## OpenAI cloud

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL=https://api.openai.com/v1`, `LLM_MODEL=gpt-4o-mini` (or similar).

## Per-role models

- **Default:** `LLM_MODEL` is used for planner, executor, critic, and final unless overridden.
- **Env:** `LLM_MODEL_PLANNER`, `LLM_MODEL_EXECUTOR`, `LLM_MODEL_CRITIC`, `LLM_MODEL_FINAL` (optional).
- **CLI:** `--model NAME` sets the default; `--model-planner`, `--model-executor`, `--model-critic`, `--model-final` set one role.
- **REPL:** `/model qwen2.5:7b` sets default; `/model planner qwen2.5:3b` sets planner only; `/model reset planner` clears that override.

All roles use the same `OPENAI_BASE_URL` and API key; only the model id string differs.

## Flags

- `--no-critic` - skip critic; final step uses plan + draft only.
- `--no-final` - stop after executor (or critic if enabled).
- `--device auto|cpu|gpu` - Ollama hint.
- `--model-planner`, `--model-executor`, `--model-critic`, `--model-final` - per-role model id.
- `--once "text"` - one run then exit (useful in scripts).

## Difference from one LLM call

Four structured stages with shared context in code: plan, draft, review, merged final output.

## Role prompts (system prompts)

Each step uses a **system prompt** (role instruction) loaded from text files in `mini_agents/prompts/`:

| File | Agent |
|------|--------|
| `planner.txt` | Agent A (planner) |
| `executor.txt` | Agent B (executor draft) |
| `critic.txt` | Critic |
| `final.txt` | Final answer |

In the REPL: `/roles` lists paths; `/role show planner` prints the file; `/role edit planner` opens the file in your default editor (Windows) or `$EDITOR` (Unix). The next user question picks up saved changes.

Shipped prompts try to reduce common failures: planner leaking the final deliverable, language drift (e.g. Russian question, English answer), incomplete CRUD lists, and chat-template garbage. Tune them for your models.

## Terminal output

Section **titles** use color (via `colorama`): planner cyan, executor blue, critic red, final green, prompt magenta. **Answer text** is plain (default terminal color). After each full run, extra blank lines separate the next prompt from the output.

- `NO_COLOR=1` (any non-empty value) disables colors.
- `FORCE_COLOR=1` keeps ANSI codes when stdout is not a TTY (e.g. piping).
