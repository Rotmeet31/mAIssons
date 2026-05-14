"""
MediaLens Fact-Check Agent.

Same tools as the research agent, adversarial reasoning pattern:
look for evidence FOR and AGAINST a specific claim, then deliver a verdict.
"""
import json
import os
import re

from loguru import logger
from openai import OpenAI

from agent import TOOL_SCHEMAS, _execute_tool
from config import LLM_MODEL, LLM_TEMPERATURE, OPENROUTER_BASE_URL


_SYSTEM_PROMPT = """\
You are a fact-checking agent for MediaLens. The user will state a specific claim.

Your job: assess whether that claim is supported, contradicted, or unclear based on
news coverage from across the political spectrum in the database.

Steps you must follow:
1. Call get_database_stats() — check how fresh and large the database is.
2. Call search_stories() — find clusters related to the claim topic.
3. Call get_lean_breakdown() on the most relevant cluster(s) — see what each political
   lean says about it. Look explicitly for articles that SUPPORT and CONTRADICT the claim.
4. If needed, call get_story_detail() for body snippets to verify specific wording.
5. Deliver a structured verdict.

Verdict — write exactly one of these in bold on the very first line of your response:
  **CONFIRMED**    — multiple sources across leans report the claim as factual
  **DISPUTED**     — sources contradict each other on this specific claim
  **MISLEADING**   — claim is technically reported but missing context that changes its meaning
  **UNVERIFIABLE** — insufficient coverage in the database to assess this claim

Structure your response exactly like this:
**<VERDICT>** — one-sentence explanation.

**Evidence for the claim**
- (cite outlet name + headline for each supporting article)

**Evidence against or complicating context**
- (cite outlet name + headline for each contradicting or contextualising article)

**What each side emphasises**
- Left: ...
- Center: ...
- Right: ...

**Confidence**
(Is the database coverage sufficient? How many sources address this directly?)

Rules:
- Cite specific outlet names and article headlines — never generic statements.
- If you cannot find relevant coverage, return UNVERIFIABLE and explain what is missing.
- Do not invent evidence. Only use what the tools return.
- Do not conflate topic coverage with claim verification — an article about immigration
  does not confirm a specific statistic about immigration unless it states it explicitly.
"""

_VERDICT_RE = re.compile(
    r"\*\*(CONFIRMED|DISPUTED|MISLEADING|UNVERIFIABLE)\*\*", re.IGNORECASE
)


def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY is not set. Add it to your .env file.")
    return OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)


def check(claim: str) -> dict:
    """
    Run the fact-check agent loop against a specific claim.

    Returns: { verdict, response, tools_called, error }
    verdict is one of: "confirmed", "disputed", "misleading", "unverifiable", None
    """
    try:
        client = _get_client()
    except EnvironmentError as exc:
        return {"verdict": None, "response": "", "tools_called": [], "error": str(exc)}

    messages: list[dict] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": claim},
    ]
    tools_called: list[str] = []

    logger.info("Fact-check claim: {!r}", claim)

    for iteration in range(10):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=LLM_TEMPERATURE,
        )

        choice = response.choices[0]
        logger.debug("Iteration {}: finish_reason={}", iteration + 1, choice.finish_reason)

        if choice.finish_reason == "stop":
            text = choice.message.content or ""
            match = _VERDICT_RE.search(text)
            verdict = match.group(1).lower() if match else None
            logger.info("Fact-check done. Verdict: {}. Tools: {}", verdict, tools_called)
            return {"verdict": verdict, "response": text, "tools_called": tools_called, "error": None}

        if choice.finish_reason != "tool_calls":
            break

        messages.append(choice.message)
        for tool_call in choice.message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            tools_called.append(name)
            logger.info("  → {}({})", name, args)
            result = _execute_tool(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result, default=str),
            })

    return {
        "verdict": None,
        "response": "",
        "tools_called": tools_called,
        "error": "Agent loop ended without a final response.",
    }
