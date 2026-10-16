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


_FENCE_LINE = re.compile(r"^\s*(```|~~~).*$")
_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*(.*?)\s*#*\s*$")
_TABLE_DIVIDER = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")
_RULE = re.compile(r"^\s*([-*_]\s*){3,}$")
_BULLET = re.compile(r"^(\s*)[*•▪●◦]\s+")
_BOLD = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_STAR = re.compile(r"(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])")
_INLINE_CODE = re.compile(r"`([^`\n]+)`")
_LINK = re.compile(r"\[([^\]\n]+)\]\((https?://[^)\s]+)\)")
_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?")


def to_plain_text(text: str) -> str:
    """Strip Markdown and other markup so the result is plain, human-readable text.

    Headings become plain lines, bold/italic/code markers are removed, links keep their
    visible text and URL, tables become simple lines, and bullets use a plain hyphen.
    """
    cleaned: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        if _FENCE_LINE.match(line) or _TABLE_DIVIDER.match(line) or _RULE.match(line):
            continue
        heading = _HEADING.match(line)
        if heading:
            line = heading.group(1)
            if line and not line.endswith((":", ".", "?", "!")):
                line += ":"
        line = _BLOCKQUOTE.sub("", line)
        line = _BULLET.sub(r"\1- ", line)
        if line.count("|") >= 2 and line.strip().startswith("|"):
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            line = " - ".join(cell for cell in cells if cell)
        line = _LINK.sub(r"\1 (\2)", line)
        line = _BOLD.sub(r"\2", line)
        line = _ITALIC_STAR.sub(r"\1", line)
        line = _INLINE_CODE.sub(r"\1", line)
        cleaned.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned)).strip()


def response_text(response) -> str:
    """Normalize a LangChain AI message into plain text and reject empty results."""
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

OUTPUT FORMAT (strict, applies to everything you write):
- Write the prompt as plain, human-readable text, the way a person would write a clear message or brief.
- Use ordinary sentences and short paragraphs. Where structure helps, use short plain-text section labels
  on their own line (for example "Goal:" or "Requirements:") followed by sentences, or simple numbered
  lines ("1.", "2.") or hyphen lines ("- ") for lists.
- Do NOT use Markdown or any markup: no # headings, no ** or * emphasis, no backticks or code fences,
  no tables, no horizontal rules, no blockquotes, no HTML or XML tags.
- Do NOT output JSON, YAML, or any other machine-readable structure, and do not wrap the answer in quotes
  or code blocks. The only exception is when the user's original request explicitly asks for a specific
  format (for example "give me the answer as JSON"); in that case, write the prompt in plain text and
  state that required format as a plain sentence inside the prompt.
- Write file names, commands, and short code identifiers inline as ordinary text, without special markers.
- Do not add a preamble, a closing remark, or any note about how the prompt was improved.
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
  request/response fields, loading states, and errors where applicable. Describe these in plain
  sentences or simple lists, not as JSON or code blocks. Never invent a route.
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

Your final answer must be plain, human-readable text only, with no Markdown or other markup, and in no
other format unless the user's original request explicitly asked for one. If the draft contains any
Markdown symbols, JSON, tables, or code fences, rewrite them as ordinary sentences, plain section labels,
and simple numbered or hyphen lists. Do not claim tests were run when they were not, and do not request
hidden chain-of-thought. Return only the final prompt, without a preamble or self-evaluation.

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
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
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
