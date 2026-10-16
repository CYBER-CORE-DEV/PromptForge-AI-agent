"""PromptForge: single-file FastAPI server + three-agent Groq/LangGraph engine.

Run locally:
    python server.py

Required environment variable:
    GROQ_API_KEY=your_real_key

Optional environment variables:
    GROQ_MODEL=openai/gpt-oss-120b
    MAX_PROMPT_LENGTH=8000
    MAX_OUTPUT_TOKENS=3000
    FRONTEND_ORIGINS=http://localhost:5500,http://127.0.0.1:5500
    HOST=127.0.0.1
    PORT=8000

The Python API server and all three agents live in this one file. A .env file is
optional but recommended for local development; API keys must never be put in frontend code.

Output format: every stage returns plain, human-readable text (no Markdown symbols).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import TypedDict

# Load a .env located beside this script, independent of the current working directory.
try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=Path(__file__).resolve().with_name(".env"))
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'python-dotenv'. Install the backend packages with:\n"
        '  python -m pip install "fastapi" "uvicorn[standard]" langchain-groq langgraph python-dotenv'
    ) from exc

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.concurrency import run_in_threadpool
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel, ConfigDict, Field, field_validator
    from langchain_groq import ChatGroq
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. Install the backend packages with:\n"
        '  python -m pip install "fastapi" "uvicorn[standard]" langchain-groq langgraph python-dotenv'
    ) from exc


# ---------------------------------------------------------------------------
# Configuration and logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("promptforge")


class ConfigurationError(RuntimeError):
    """Raised when server configuration is missing or invalid."""


class PromptValidationError(ValueError):
    """Raised when a prompt fails validation inside the enhancement engine."""


def positive_int_setting(name: str, default: int) -> int:
    """Read a positive integer environment setting with an actionable error."""
    raw_value = os.getenv(name, str(default)).strip()
    try:
        value = int(raw_value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            f"{name} must be a positive integer; received {raw_value!r}."
        ) from exc
    if value < 1:
        raise ConfigurationError(f"{name} must be greater than zero.")
    return value


MAX_PROMPT_LENGTH = positive_int_setting("MAX_PROMPT_LENGTH", 8000)
MAX_OUTPUT_TOKENS = positive_int_setting("MAX_OUTPUT_TOKENS", 3000)
DEFAULT_MODEL = "openai/gpt-oss-120b"


def configured_model() -> str:
    return os.getenv("GROQ_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def allowed_frontend_origins() -> list[str]:
    """Use explicit configured origins, or practical localhost defaults for development."""
    raw = os.getenv("FRONTEND_ORIGINS", "").strip()
    if raw:
        return [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


# ---------------------------------------------------------------------------
# Plain-text clean-up (safety net for the "human readable plain format" rule)
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^[ \t]*(```|~~~)[^\n]*\n?", re.M)
_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]*", re.M)
_RULE = re.compile(r"^[ \t]*([-*_])([ \t]*\1){2,}[ \t]*$", re.M)
_TABLE_SEP = re.compile(r"^[ \t]*\|?[ \t]*:?-{3,}:?[ \t]*(\|[ \t]*:?-{3,}:?[ \t]*)*\|?[ \t]*$", re.M)
_QUOTE = re.compile(r"^[ \t]*>[ \t]?", re.M)
_BULLET = re.compile(r"^([ \t]*)[*\u2022+][ \t]+", re.M)
_BOLD = re.compile(r"\*\*(.+?)\*\*")
_ITALIC = re.compile(r"(?<![\w*])\*(?![\s*])([^*\n]+?)(?<![\s*])\*(?![\w*])")
_STRIKE = re.compile(r"~~(.+?)~~")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BLANKS = re.compile(r"\n{3,}")


def to_plain_text(text: str) -> str:
    """Strip Markdown syntax so the result reads as ordinary plain text.

    The agents are instructed to write plain text already; this guarantees the
    final output stays clean even if a model slips back into Markdown habits.
    """
    text = text.replace("\r\n", "\n")
    text = _FENCE.sub("", text)
    text = _TABLE_SEP.sub("", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.startswith("|") and stripped.count("|") >= 2:
            cells = [cell.strip() for cell in stripped.strip("|").split("|")]
            line = "- " + "; ".join(cell for cell in cells if cell)
        lines.append(line)
    text = "\n".join(lines)
    text = _HEADING.sub("", text)
    text = _RULE.sub("", text)
    text = _QUOTE.sub("", text)
    text = _BULLET.sub(r"\1- ", text)
    text = _LINK.sub(r"\1 (\2)", text)
    text = _BOLD.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _STRIKE.sub(r"\1", text)
    text = text.replace("`", "")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return _BLANKS.sub("\n\n", text).strip()


# ---------------------------------------------------------------------------
# Groq model and three-agent LangGraph workflow
# ---------------------------------------------------------------------------

_llm = None


def get_llm():
    """Initialize the model on first request, rather than blocking server startup."""
    global _llm
    if _llm is not None:
        return _llm

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key or api_key == "your_groq_api_key_here":
        raise ConfigurationError(
            "GROQ_API_KEY is missing or still uses the example placeholder. "
            "Add your real Groq key to the .env file beside server.py."
        )

    # Keep provider settings conservative. No reasoning_format argument is passed here.
    _llm = ChatGroq(
        model=configured_model(),
        temperature=0.3,
        api_key=api_key,
        timeout=60,
        max_retries=2,
        max_tokens=MAX_OUTPUT_TOKENS,
    )
    return _llm


def response_text(response) -> str:
    """Normalize a LangChain AI message into clean plain text and reject empty results."""
    content = getattr(response, "content", response)
    if content is None:
        result = ""
    elif isinstance(content, str):
        result = content.strip()
    elif isinstance(content, list):
        pieces: list[str] = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
            else:
                text_part = getattr(item, "text", None)
                if isinstance(text_part, str):
                    pieces.append(text_part)
        result = "\n".join(part.strip() for part in pieces if part.strip()).strip()
    else:
        result = str(content).strip()

    result = to_plain_text(result)
    if not result or result.lower() == "none":
        raise RuntimeError("The model returned an empty response.")
    return result


class PromptState(TypedDict):
    original_prompt: str
    agent1_prompt: str
    agent2_prompt: str
    agent3_prompt: str


SHARED_RULES = """
General rules for improving the user's prompt:
- Preserve the user's actual goal, explicit constraints, language, tools, and intended audience.
- Improve clarity and usefulness; do not turn the request into a different or unnecessarily larger task.
- Never invent project details, existing features, API routes, filenames, credentials, data, or test results.
- Distinguish supplied facts from assumptions. Use clear placeholders for genuinely unknown values.
- Keep simple requests concise and complex requests as detailed as necessary.
- Prefer concrete, testable acceptance criteria over vague claims such as "perfect" or "10x better".
- Do not request hidden chain-of-thought. Ask for concise explanations, assumptions, and verification steps.
- Avoid repeated instructions, unrelated features, and unnecessary dependencies.
- Return only the revised prompt, without commentary about the revision.

Output format (strict): write in plain, human-readable text, the way a thoughtful person would write
a clear message to a colleague.
- Do not use Markdown or any markup: no # headings, no ** or * for emphasis, no backticks or code
  fences, no tables or pipe characters, no horizontal rules, no quote markers, no emojis.
- Use short paragraphs separated by one blank line.
- For sections, write a short plain label on its own line ending with a colon, for example
  "Goal:", "Context:", "Requirements:", "Output format:", then the content on the lines below it.
- For lists, put each item on its own line starting with "1.", "2.", "3." for ordered steps, or
  "- " for unordered items. Keep each item to one or two sentences.
- Use JSON, code, or any other format only when the user's request explicitly asks for it.
""".strip()


def agent_1_structural_foundation(state: PromptState) -> PromptState:
    """Agent 1: Define the objective, context, constraints, and deliverable."""
    system_prompt = f"""
You are Agent 1: Prompt Architect.

Transform the user's request into a clear, faithful, self-contained prompt. Include only the relevant
items: objective and deliverable, supplied context/materials, intended audience, requirements,
constraints, expected output format, and checkable acceptance criteria.

Preserve explicit technology preferences and exact-output requirements. Treat supplied code and logs as
the source of truth. Do not assume a command-line Python script already exposes an HTTP API; if a
frontend needs one, explicitly require an API layer to be built. Do not solve the underlying task;
create the prompt another model will use.

{SHARED_RULES}
""".strip()
    response = get_llm().invoke([
        ("system", system_prompt),
        ("user", f"Original user request:\n{state['original_prompt']}"),
    ])
    state["agent1_prompt"] = response_text(response)
    return state


def agent_2_depth_and_edge_cases(state: PromptState) -> PromptState:
    """Agent 2: Add relevant implementation details, edge cases, and tests."""
    system_prompt = f"""
You are Agent 2: Requirements and Edge-Case Specialist.

Refine Agent 1's prompt without changing its intent. Add task-relevant specifics about:
- Expected inputs, processing, outputs, and important state transitions.
- Integration contracts only when known. For frontend/backend tasks, specify HTTP method,
  request/response JSON shapes, loading states, and errors where applicable. Never invent a route.
- Relevant validation, empty input, malformed responses, timeout, network failure, and security cases.
- Practical tests for the main flow and important failure paths.
- Observable acceptance criteria. Do not claim tests have already passed without evidence.

Keep the detail proportional to the task. Do not add unrelated features, duplicate sections, or require
frameworks/file trees/JSON unless useful or explicitly requested.

{SHARED_RULES}
""".strip()
    response = get_llm().invoke([
        ("system", system_prompt),
        ("user", f"Prompt to refine:\n{state['agent1_prompt']}"),
    ])
    state["agent2_prompt"] = response_text(response)
    return state


def agent_3_formatting_and_precision(state: PromptState) -> PromptState:
    """Agent 3: Remove duplication and produce the final ready-to-use prompt."""
    system_prompt = f"""
You are Agent 3: Final Prompt Editor and Quality Reviewer.

Produce the final prompt from Agent 2's draft. Do not inflate it just to appear comprehensive. Ensure it
preserves the user's goal, constraints, and deliverable; distinguishes known facts from assumptions;
contains relevant, actionable requirements; includes acceptance criteria where useful; and has no
contradictions, invented capabilities, or duplicated rules.

Your final output must read naturally to a person: plain text only, with clear short paragraphs, simple
labeled sections, and simple numbered or dashed lists, exactly as described in the output format rules
below. Remove any Markdown symbols left over from earlier drafts. Do not claim tests were run when they
were not, and do not request hidden chain-of-thought. Return only the final prompt, without a preamble
or self-evaluation.

{SHARED_RULES}
""".strip()
    response = get_llm().invoke([
        ("system", system_prompt),
        ("user", f"Draft to finalize:\n{state['agent2_prompt']}"),
    ])
    state["agent3_prompt"] = response_text(response)
    return state


# Compiled once on import. Model calls happen only when /enhance is requested.
workflow = StateGraph(PromptState)
workflow.add_node("agent_1", agent_1_structural_foundation)
workflow.add_node("agent_2", agent_2_depth_and_edge_cases)
workflow.add_node("agent_3", agent_3_formatting_and_precision)
workflow.add_edge(START, "agent_1")
workflow.add_edge("agent_1", "agent_2")
workflow.add_edge("agent_2", "agent_3")
workflow.add_edge("agent_3", END)
workflow_app = workflow.compile()


def enhance_prompt(user_input: str) -> dict[str, str]:
    """Run all three agents and return the three outputs."""
    if not isinstance(user_input, str) or not user_input.strip():
        raise PromptValidationError("Prompt cannot be empty.")

    prompt = user_input.strip()
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise PromptValidationError(
            f"Prompt is too long ({len(prompt)} characters). Maximum is {MAX_PROMPT_LENGTH}."
        )

    initial_state: PromptState = {
        "original_prompt": prompt,
        "agent1_prompt": "",
        "agent2_prompt": "",
        "agent3_prompt": "",
    }
    result = workflow_app.invoke(initial_state)
    final_prompt = result.get("agent3_prompt", "").strip()
    if not final_prompt:
        raise RuntimeError("The workflow completed without a final prompt.")

    return {
        "agent1_prompt": result.get("agent1_prompt", "").strip(),
        "agent2_prompt": result.get("agent2_prompt", "").strip(),
        "agent3_prompt": final_prompt,
    }


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

class EnhanceRequest(BaseModel):
    """JSON body accepted by POST /enhance."""

    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=1, max_length=MAX_PROMPT_LENGTH)

    @field_validator("prompt")
    @classmethod
    def prompt_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty or whitespace only.")
        return cleaned


class PromptStages(BaseModel):
    structural: str
    depth_and_edge_cases: str
    formatting_and_precision: str


class EnhanceResponse(BaseModel):
    enhanced_prompt: str
    stages: PromptStages
    word_count: int


class HealthResponse(BaseModel):
    status: str
    service: str
    model: str
    api_key_configured: bool


app = FastAPI(
    title="PromptForge API",
    description=(
        "Single-file HTTP server for PromptForge's three-agent prompt enhancer. "
        "POST JSON to /enhance to receive the final prompt and intermediate stages."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_frontend_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/", tags=["general"])
def root() -> dict[str, str]:
    return {
        "service": "PromptForge API",
        "status": "ok",
        "health": "/health",
        "enhance": "POST /enhance",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["general"])
def health() -> HealthResponse:
    """Check server/configuration status without making a model request."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    configured = bool(key and key != "your_groq_api_key_here")
    return HealthResponse(
        status="ready" if configured else "setup_required",
        service="PromptForge API",
        model=configured_model(),
        api_key_configured=configured,
    )


@app.post("/enhance", response_model=EnhanceResponse, tags=["prompt enhancement"])
async def enhance(request: EnhanceRequest) -> EnhanceResponse:
    """Run the synchronous LangGraph/model workflow without blocking the async event loop."""
    try:
        result = await run_in_threadpool(enhance_prompt, request.prompt)
        final_prompt = result["agent3_prompt"].strip()
        if not final_prompt:
            raise RuntimeError("The workflow returned an empty final prompt.")
        return EnhanceResponse(
            enhanced_prompt=final_prompt,
            stages=PromptStages(
                structural=result.get("agent1_prompt", ""),
                depth_and_edge_cases=result.get("agent2_prompt", ""),
                formatting_and_precision=final_prompt,
            ),
            word_count=len(final_prompt.split()),
        )
    except PromptValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ConfigurationError as exc:
        logger.warning("PromptForge configuration issue: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Prompt enhancement is not configured. Check the server's GROQ_API_KEY and settings.",
        ) from exc
    except Exception as exc:
        # Details stay in server logs, not in public API responses.
        logger.exception("Prompt enhancement failed (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=502,
            detail="The prompt provider failed to complete the request. Please retry shortly.",
        ) from exc


if __name__ == "__main__":
    import uvicorn

    try:
        port = positive_int_setting("PORT", 8000)
    except ConfigurationError as exc:
        raise SystemExit(f"Server configuration error: {exc}") from exc

    host = os.getenv("HOST", "127.0.0.1").strip() or "127.0.0.1"
    logger.info("Starting PromptForge API on http://%s:%s", host, port)
    uvicorn.run(app, host=host, port=port)
