"""
Shared utilities for phase-locked agent orchestrators.

Both agent.py and fact_check.py import run_tool_phase() from here
instead of duplicating the tool-calling loop logic.
"""
from __future__ import annotations

import json
import sys
import os
from typing import Any, Callable

from loguru import logger

sys.path.insert(0, os.path.dirname(__file__))
from config import LLM_MODEL, LLM_TEMPERATURE
from schemas import RunTrace


def run_tool_phase(
    client: Any,
    trace: RunTrace,
    system: str,
    user_content: str,
    allowed_tool_names: set[str],
    required_tool_names: set[str],
    required_fallback_args: dict[str, dict[str, Any]],
    tool_schemas: list[dict[str, Any]],
    tool_registry: dict[str, Callable[..., Any]],
    max_iters: int = 5,
) -> dict[str, Any]:
    """
    Run one phase of the agent loop.

    Calls the LLM with only the tools listed in allowed_tool_names.
    After the loop, any required tool not called by the LLM is called
    programmatically using required_fallback_args.

    Returns a dict mapping tool_name → last result returned by that tool.
    """
    filtered_schemas = [
        s for s in tool_schemas
        if s["function"]["name"] in allowed_tool_names
    ]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    results: dict[str, Any] = {}
    called: set[str] = set()

    for _ in range(max_iters):
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=filtered_schemas,
            tool_choice="auto",
            temperature=LLM_TEMPERATURE,
        )
        choice = response.choices[0]
        usage = response.usage

        if choice.finish_reason in ("stop", "length"):
            break
        if choice.finish_reason != "tool_calls":
            break

        messages.append(choice.message)
        for tc in choice.message.tool_calls:
            name = tc.function.name
            if name not in allowed_tool_names:
                logger.warning("Phase ignored out-of-scope tool call: {}", name)
                continue
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}

            fn = tool_registry.get(name)
            if fn is None:
                result: Any = {"error": f"unknown tool: {name}"}
            else:
                try:
                    result = fn(**args)
                except Exception as exc:
                    result = {"error": str(exc)}

            results[name] = result
            called.add(name)
            trace.add_step(
                name, args, result,
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    # Enforce required tools that the LLM skipped
    for tool_name in required_tool_names - called:
        fallback_args = required_fallback_args.get(tool_name, {})
        fn = tool_registry.get(tool_name)
        if fn is None:
            logger.error("Cannot enforce required tool {} — not in registry", tool_name)
            continue
        try:
            result = fn(**fallback_args)
        except Exception as exc:
            result = {"error": str(exc)}
        results[tool_name] = result
        trace.add_step(tool_name, fallback_args, result)
        logger.warning(
            "Phase enforced mandatory tool call: {}({})", tool_name, fallback_args
        )

    return results
