"""
Orchestration: planner -> executor -> critic (optional) -> final answer.
OpenAI API or Ollama OpenAI-compatible endpoint. Interactive REPL by default.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

try:
    from mini_agents.roles import (
        ensure_prompt_files,
        load_system_prompt,
        open_prompt_editor,
        prompt_path,
        valid_roles,
    )
except ImportError:
    from roles import (
        ensure_prompt_files,
        load_system_prompt,
        open_prompt_editor,
        prompt_path,
        valid_roles,
    )

_TERM: Any = None


def _term() -> Any:
    """Lazy terminal theme (colorama). Respects NO_COLOR; FORCE_COLOR=1 for pipes."""
    global _TERM
    if _TERM is not None:
        return _TERM

    class T:
        reset = ""
        dim = ""
        rule = ""
        plan_h = ""
        exec_h = ""
        critic_h = ""
        final_h = ""
        alt_h = ""
        prompt = "mini-agents> "
        ok = muted = err = ""
        cmd = desc = ""
        banner_hi = ""

    t = T()
    _TERM = t
    if os.getenv("NO_COLOR", "").strip():
        return t
    try:
        from colorama import Fore, Style, init

        force = os.getenv("FORCE_COLOR", "").lower() in ("1", "true", "yes")
        strip = (not sys.stdout.isatty()) and not force
        init(autoreset=True, strip=strip)
        R = Style.RESET_ALL
        D = Style.DIM
        B = Style.BRIGHT
        t.reset = R
        t.dim = D
        t.rule = D + ("\u2500" * 56) + R
        t.plan_h = B + Fore.CYAN
        t.exec_h = B + Fore.BLUE
        t.critic_h = B + Fore.RED
        t.final_h = B + Fore.GREEN
        t.alt_h = B + Fore.MAGENTA
        t.prompt = B + Fore.MAGENTA + "mini-agents> " + R
        t.ok = Fore.GREEN
        t.muted = D + Fore.WHITE
        t.err = Fore.RED
        t.cmd = B + Fore.CYAN
        t.desc = D
        t.banner_hi = B + Fore.CYAN
    except ImportError:
        pass
    return t


def _init_terminal() -> None:
    _term()


def _print_rule(t: Any) -> None:
    if t.rule:
        print(t.rule, flush=True)
    else:
        print("-" * 56, flush=True)


def _print_plain_body(text: str) -> None:
    """Answer text without ANSI (easier to read)."""
    print(text + "\n", flush=True)


def _section_header(
    t: Any, color_attr: str, title: str, *, leading_blank: bool = True
) -> None:
    if leading_blank:
        print()
    _print_rule(t)
    print()
    hc = getattr(t, color_attr)
    print(hc + title + t.reset, flush=True)
    print()


def _print_after_turn(t: Any) -> None:
    """Breathing room before the next prompt or shell return."""
    print(t.reset + "\n\n", flush=True)


def _load_env() -> None:
    pkg = Path(__file__).resolve().parent
    load_dotenv(pkg / ".env")
    load_dotenv()


def _configure_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass


def _bool_env(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _is_ollama_base_url(url: str) -> bool:
    u = url.lower().rstrip("/")
    return "11434" in u or (u.endswith("/v1") and ("localhost" in u or "127.0.0.1" in u))


MODEL_ROLE_KEYS = ("planner", "executor", "critic", "final")


@dataclass
class Session:
    """model = default for all steps; model_overrides[role] replaces it per agent."""

    model: str
    model_overrides: dict[str, str] = field(default_factory=dict)
    device: str = "auto"
    critic_enabled: bool = True
    final_enabled: bool = True
    base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    )

    def model_for(self, role: str) -> str:
        if role in self.model_overrides:
            return self.model_overrides[role]
        return self.model


def _client_for_session(session: Session) -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or "ollama"
    return OpenAI(api_key=api_key, base_url=session.base_url)


def _completion_extra_body(session: Session) -> dict[str, Any] | None:
    if not _is_ollama_base_url(session.base_url):
        return None
    if session.device == "auto":
        return None
    if session.device == "cpu":
        return {"options": {"num_gpu": 0}}
    if session.device == "gpu":
        return {"options": {"num_gpu": 999}}
    return None


def _chat_create(
    client: OpenAI,
    session: Session,
    messages: list[dict[str, str]],
    *,
    model_name: str,
) -> Any:
    extra = _completion_extra_body(session)
    kwargs: dict[str, Any] = {"model": model_name, "messages": messages}
    if extra:
        kwargs["extra_body"] = extra
    try:
        return client.chat.completions.create(**kwargs)
    except Exception:
        if extra:
            return client.chat.completions.create(model=model_name, messages=messages)
        raise


def run_planner(client: OpenAI, session: Session, user_question: str) -> str:
    messages = [
        {"role": "system", "content": load_system_prompt("planner")},
        {"role": "user", "content": user_question},
    ]
    r = _chat_create(
        client, session, messages, model_name=session.model_for("planner")
    )
    return (r.choices[0].message.content or "").strip()


def run_executor(client: OpenAI, session: Session, user_question: str, plan: str) -> str:
    messages = [
        {"role": "system", "content": load_system_prompt("executor")},
        {
            "role": "user",
            "content": f"Original request:\n{user_question}\n\nPlan from planner:\n{plan}",
        },
    ]
    r = _chat_create(
        client, session, messages, model_name=session.model_for("executor")
    )
    return (r.choices[0].message.content or "").strip()


def run_critic(
    client: OpenAI,
    session: Session,
    user_question: str,
    plan: str,
    answer: str,
) -> str:
    messages = [
        {"role": "system", "content": load_system_prompt("critic")},
        {
            "role": "user",
            "content": (
                f"Request:\n{user_question}\n\nPlan:\n{plan}\n\nDraft answer:\n{answer}"
            ),
        },
    ]
    r = _chat_create(
        client, session, messages, model_name=session.model_for("critic")
    )
    return (r.choices[0].message.content or "").strip()


def run_final(
    client: OpenAI,
    session: Session,
    user_question: str,
    plan: str,
    draft: str,
    critique: str,
    critic_ran: bool,
) -> str:
    if critic_ran:
        critic_section = f"Critic review:\n{critique}"
        orch = (
            "\n\nOrchestration note: If the critic listed concrete issues, fix them in your final answer. "
            "If the critic replied with only OK, keep the draft except for tiny clarity fixes."
        )
    else:
        critic_section = "(No critic step was run.)"
        orch = (
            "\n\nOrchestration note: No critic step was run. Produce the best single answer "
            "for the user using the plan and draft only."
        )
    messages = [
        {
            "role": "system",
            "content": load_system_prompt("final") + orch,
        },
        {
            "role": "user",
            "content": (
                f"User request:\n{user_question}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Executor draft:\n{draft}\n\n"
                f"{critic_section}"
            ),
        },
    ]
    r = _chat_create(
        client, session, messages, model_name=session.model_for("final")
    )
    return (r.choices[0].message.content or "").strip()


def run_pipeline(client: OpenAI, session: Session, user_question: str) -> None:
    t = _term()
    print()
    _section_header(t, "plan_h", "Agent A · planner", leading_blank=False)
    plan = run_planner(client, session, user_question)
    _print_plain_body(plan)

    _section_header(t, "exec_h", "Agent B · executor (draft)")
    draft = run_executor(client, session, user_question, plan)
    _print_plain_body(draft)

    critique = ""
    critic_ran = False
    if session.critic_enabled:
        _section_header(t, "critic_h", "Critic")
        critique = run_critic(client, session, user_question, plan, draft)
        _print_plain_body(critique)
        critic_ran = True

    if session.final_enabled:
        _section_header(t, "final_h", "Final answer")
        final_text = run_final(
            client, session, user_question, plan, draft, critique, critic_ran
        )
        _print_plain_body(final_text)
    else:
        _section_header(t, "alt_h", "Final (draft only, final step off)")
        _print_plain_body(draft)

    _print_after_turn(t)


def print_slash_help() -> None:
    t = _term()
    text = """
Slash commands (interactive) or: python -m mini_agents /help
  /help              This text
  /quit  /exit       Leave interactive mode
  /model NAME        Set default model for all roles without a per-role override
  /model ROLE NAME   Per-role model (ROLE: planner|executor|critic|final), e.g. /model planner qwen2.5:3b
  /model reset ROLE  Drop per-role override (ROLE as above)
  /device auto|cpu|gpu   Ollama hint (see README); cpu uses extra_body options when supported
  /critic on|off     Enable or disable critic (final still uses draft if critic off)
  /final on|off      Enable or disable final synthesis step
  /status            model, device, critic, final, base URL
  /models            Run `ollama list` if ollama is on PATH
  /roles             List roles and paths to system prompt files (prompts/*.txt)
  /role              Help for editing roles
  /role show NAME    Print prompt (planner|executor|critic|final)
  /role edit NAME    Open prompt file (Windows: default app; Unix: $EDITOR)
  /role path NAME    Print file path only

System prompts (role instructions) live under mini_agents/prompts/. Changes apply on the next question.

One-shot: mini-agents "your question"   or   python -m mini_agents "question"
GPU/CPU (Ollama server): for CPU-only restart with OLLAMA_NUM_GPU=0 before ollama serve (see README).
Colors: set NO_COLOR=1 to disable. FORCE_COLOR=1 forces ANSI when stdout is not a TTY.
"""
    print(t.banner_hi + "Help" + t.reset)
    print(t.desc + text.strip() + t.reset + "\n", flush=True)


def cmd_models() -> None:
    exe = shutil.which("ollama")
    if not exe:
        print("ollama CLI not found on PATH.")
        return
    try:
        out = subprocess.run(
            [exe, "list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
        )
        if out.stdout:
            print(out.stdout.rstrip())
        if out.stderr:
            print(out.stderr.rstrip(), file=sys.stderr)
        if out.returncode != 0:
            print(f"(exit code {out.returncode})", file=sys.stderr)
    except OSError as e:
        print(f"Could not run ollama: {e}", file=sys.stderr)


def cmd_status(session: Session) -> None:
    t = _term()
    on = t.ok + "on" + t.reset
    off = t.muted + "off" + t.reset
    print(t.banner_hi + "Status" + t.reset)
    print(f"{t.muted}default model{t.reset} {session.model}")
    for rk in MODEL_ROLE_KEYS:
        m = session.model_for(rk)
        tag = f" {t.ok}*{t.reset}" if rk in session.model_overrides else ""
        print(f"{t.muted}model {rk}{t.reset}  {m}{tag}")
    print(f"{t.muted}device{t.reset}        {session.device}")
    print(f"{t.muted}critic{t.reset}        {on if session.critic_enabled else off}")
    print(f"{t.muted}final{t.reset}         {on if session.final_enabled else off}")
    print(f"{t.muted}base_url{t.reset}      {session.base_url}\n", flush=True)


def handle_slash_line(line: str, session: Session) -> bool:
    s = line.strip()
    if not s.startswith("/"):
        return True

    parts = s.split(maxsplit=1)
    cmd = parts[0].lower()
    arg = parts[1].strip() if len(parts) > 1 else ""

    if cmd in ("/help", "/?"):
        print_slash_help()
        return False
    if cmd in ("/quit", "/exit"):
        raise EOFError
    if cmd == "/model":
        t = _term()
        if not arg:
            print(
                t.err
                + "Usage: /model <default-name>  |  /model <role> <name>  |  /model reset <role>"
                + t.reset
            )
            print(
                t.desc
                + f"roles: {', '.join(MODEL_ROLE_KEYS)}"
                + t.reset,
                flush=True,
            )
            return False
        parts = arg.split(maxsplit=1)
        if parts[0].lower() == "reset":
            if len(parts) < 2:
                print(t.err + "Usage: /model reset <role>" + t.reset)
                return False
            role = parts[1].strip().lower()
            if role not in MODEL_ROLE_KEYS:
                print(
                    t.err + f"Unknown role: {role}. Use: {', '.join(MODEL_ROLE_KEYS)}" + t.reset
                )
                return False
            session.model_overrides.pop(role, None)
            print(
                t.ok
                + f"{role} now uses default model: {session.model}"
                + t.reset,
                flush=True,
            )
            return False
        if len(parts) == 1:
            session.model = parts[0].strip()
            print(t.ok + f"default model: {session.model}" + t.reset, flush=True)
            return False
        role, name = parts[0].lower(), parts[1].strip()
        if role not in MODEL_ROLE_KEYS:
            print(
                t.err
                + f"Unknown role '{role}'. Use: {', '.join(MODEL_ROLE_KEYS)} or one word for default."
                + t.reset
            )
            return False
        if not name:
            print(t.err + "Missing model name." + t.reset)
            return False
        session.model_overrides[role] = name
        print(
            t.ok + f"model for {role}: {name}" + t.reset,
            flush=True,
        )
        return False
    if cmd == "/device":
        t = _term()
        a = arg.lower()
        if a not in ("auto", "cpu", "gpu"):
            print(t.err + "Usage: /device auto|cpu|gpu" + t.reset)
        else:
            session.device = a
            print(t.ok + f"device set to: {session.device}" + t.reset)
        return False
    if cmd == "/critic":
        t = _term()
        a = arg.lower()
        if a in ("on", "1", "true", "yes"):
            session.critic_enabled = True
            print(t.ok + "critic: on" + t.reset)
        elif a in ("off", "0", "false", "no"):
            session.critic_enabled = False
            print(t.ok + "critic: off" + t.reset)
        else:
            print(t.err + "Usage: /critic on|off" + t.reset)
        return False
    if cmd == "/final":
        t = _term()
        a = arg.lower()
        if a in ("on", "1", "true", "yes"):
            session.final_enabled = True
            print(t.ok + "final: on" + t.reset)
        elif a in ("off", "0", "false", "no"):
            session.final_enabled = False
            print(t.ok + "final: off" + t.reset)
        else:
            print(t.err + "Usage: /final on|off" + t.reset)
        return False
    if cmd == "/status":
        cmd_status(session)
        return False
    if cmd == "/models":
        cmd_models()
        return False
    if cmd == "/roles":
        t = _term()
        print(t.banner_hi + "Roles (system prompts)" + t.reset + "\n")
        for r in valid_roles():
            p = prompt_path(r)
            print(f"  {t.cmd}{r}{t.reset}  {t.muted}{p}{t.reset}")
        print()
        return False
    if cmd == "/role":
        t = _term()
        if not arg:
            print(t.desc + "Edit system prompts (role instructions) on disk:\n" + t.reset)
            print(f"  {t.cmd}/roles{t.reset}               list roles and paths")
            print(f"  {t.cmd}/role show NAME{t.reset}     print prompt text")
            print(f"  {t.cmd}/role edit NAME{t.reset}     open in editor")
            print(f"  {t.cmd}/role path NAME{t.reset}     print path only")
            print(
                f"\n  NAME is one of: {t.ok}{', '.join(valid_roles())}{t.reset}\n",
                flush=True,
            )
            return False
        sub = arg.split(maxsplit=1)
        verb = sub[0].lower()
        tail = sub[1].strip() if len(sub) > 1 else ""
        if verb == "show":
            role = tail.lower()
            if role not in valid_roles():
                print(
                    t.err + f"Unknown role. Use: {', '.join(valid_roles())}" + t.reset
                )
                return False
            print(t.banner_hi + f"prompt: {role}" + t.reset + "\n")
            print(load_system_prompt(role) + "\n", flush=True)
            return False
        if verb == "path":
            role = tail.lower()
            if role not in valid_roles():
                print(
                    t.err + f"Unknown role. Use: {', '.join(valid_roles())}" + t.reset
                )
                return False
            print(str(prompt_path(role)) + "\n", flush=True)
            return False
        if verb == "edit":
            role = tail.lower()
            if role not in valid_roles():
                print(
                    t.err + f"Unknown role. Use: {', '.join(valid_roles())}" + t.reset
                )
                return False
            try:
                open_prompt_editor(role)
                print(
                    t.ok + f"Opened {prompt_path(role)} - save the file; next question uses new text."
                    + t.reset,
                    flush=True,
                )
            except OSError as e:
                print(t.err + str(e) + t.reset, flush=True)
            return False
        print(
            t.err + "Usage: /role show|edit|path <planner|executor|critic|final>" + t.reset
        )
        return False

    t = _term()
    print(t.err + f"Unknown command: {cmd}. Type /help" + t.reset)
    return False


def interactive_loop(session: Session, client: OpenAI) -> None:
    t = _term()
    print(
        t.banner_hi
        + "mini-agents"
        + t.reset
        + t.desc
        + "  Questions or /help. Ctrl+Z Enter (Windows) / Ctrl+D (Unix) to exit.\n"
        + t.reset,
        flush=True,
    )
    while True:
        try:
            line = input(t.prompt).strip()
        except EOFError:
            print()
            break
        if not line:
            continue
        try:
            should_run = handle_slash_line(line, session)
        except EOFError:
            break
        if not should_run:
            continue
        run_pipeline(client, session, line)


def _maybe_run_slash_or_pipeline(user_question: str, session: Session) -> None:
    """Slash lines are commands; otherwise run the LLM pipeline."""
    client = _client_for_session(session)
    if user_question.startswith("/"):
        if handle_slash_line(user_question, session):
            run_pipeline(client, session, user_question)
        return
    run_pipeline(client, session, user_question)


def _model_overrides_from_env() -> dict[str, str]:
    out: dict[str, str] = {}
    env_map = {
        "planner": "LLM_MODEL_PLANNER",
        "executor": "LLM_MODEL_EXECUTOR",
        "critic": "LLM_MODEL_CRITIC",
        "final": "LLM_MODEL_FINAL",
    }
    for role, env_key in env_map.items():
        v = os.getenv(env_key)
        if v and v.strip():
            out[role] = v.strip()
    return out


def build_session_from_args(args: argparse.Namespace) -> Session:
    model = args.model or os.getenv("LLM_MODEL", "qwen2.5:7b")
    overrides = _model_overrides_from_env()
    for role in MODEL_ROLE_KEYS:
        flag = getattr(args, f"model_{role}", None)
        if flag:
            overrides[role] = flag.strip()
    critic = _bool_env("ENABLE_CRITIC", True)
    if args.no_critic:
        critic = False
    final_step = _bool_env("ENABLE_FINAL", True)
    if getattr(args, "no_final", False):
        final_step = False
    device = (args.device or os.getenv("OLLAMA_DEVICE", "auto")).lower()
    if device not in ("auto", "cpu", "gpu"):
        device = "auto"
    base = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:11434/v1")
    return Session(
        model=model,
        model_overrides=overrides,
        device=device,
        critic_enabled=critic,
        final_enabled=final_step,
        base_url=base,
    )


def main() -> None:
    _configure_stdio()
    _load_env()
    ensure_prompt_files()
    _init_terminal()
    parser = argparse.ArgumentParser(
        description="Planner -> executor -> critic -> final answer. Default: interactive REPL.",
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="One-shot question (omit for interactive REPL when stdin is a terminal)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Force interactive mode even if arguments are present",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one pipeline from positional args then exit (non-interactive)",
    )
    parser.add_argument("--model", metavar="NAME", help="Default chat model (LLM_MODEL)")
    parser.add_argument(
        "--model-planner",
        metavar="NAME",
        help="Override LLM_MODEL_PLANNER",
    )
    parser.add_argument(
        "--model-executor",
        metavar="NAME",
        help="Override LLM_MODEL_EXECUTOR",
    )
    parser.add_argument(
        "--model-critic",
        metavar="NAME",
        help="Override LLM_MODEL_CRITIC",
    )
    parser.add_argument(
        "--model-final",
        metavar="NAME",
        help="Override LLM_MODEL_FINAL",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "gpu"),
        help="Ollama device hint; default OLLAMA_DEVICE or auto",
    )
    parser.add_argument("--no-critic", action="store_true", help="Disable critic step")
    parser.add_argument(
        "--no-final",
        action="store_true",
        help="Disable final synthesis (stop after executor or critic)",
    )
    args = parser.parse_args()

    if args.question:
        user_question = " ".join(args.question).strip()
    elif args.interactive or sys.stdin.isatty():
        # Do not block on sys.stdin.read() waiting for EOF when the user expects a REPL.
        user_question = ""
    else:
        user_question = sys.stdin.read().strip()

    session = build_session_from_args(args)

    if args.interactive:
        client = _client_for_session(session)
        interactive_loop(session, client)
        return

    if args.once:
        if not user_question:
            print("Usage: mini-agents --once \"your question\"", file=sys.stderr)
            sys.exit(1)
        if user_question == "help":
            print_slash_help()
            return
        _maybe_run_slash_or_pipeline(user_question, session)
        return

    if user_question:
        if user_question == "help":
            print_slash_help()
            return
        _maybe_run_slash_or_pipeline(user_question, session)
        return

    client = _client_for_session(session)
    interactive_loop(session, client)


if __name__ == "__main__":
    main()
