"""Test data-wrapped context compaction against the real model API.

    uv run python scripts/test_compaction.py
    uv run python scripts/test_compaction.py --model deepseek-v4-flash

Loads OPENAI_* settings from .env. Makes one API request without tools or a sandbox.
Uses saved pre-compaction history, not the current compact_context implementation.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from assignment.agent.base import (
    COMPACTION_SYSTEM_PROMPT,
    content_text,
    rough_message_tokens,
    serialize_conversation,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory", type=Path,
        default=ROOT / "artifacts/django__django-15368-trajectory.json",
    )
    parser.add_argument("--event", type=int, default=0, help="Compaction event index (default: 0)")
    parser.add_argument("--model", help="Override OPENAI_MODEL (falls back to recorded model)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--keep-recent", type=int, default=1, help="Recent complete steps to retain")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "artifacts/compaction-test.json",
    )
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")

    trajectory = json.loads(args.trajectory.read_text())
    events = trajectory.get("compactions") or []
    original_response = None
    if events:
        if not 0 <= args.event < len(events):
            parser.error(f"--event must be between 0 and {len(events) - 1}")
        event = events[args.event]
        before = event["active_prompt_before"]
        original_response = event.get("compaction_response")
    else:
        if args.event != 0:
            parser.error("No compaction events exist; omit --event to use the latest saved prompt")
        prompts = trajectory.get("prompts") or []
        if not prompts:
            parser.error("Trajectory contains neither compaction events nor saved prompts")
        before = prompts[-1]
        print("No compaction events; using the latest saved prompt instead.", flush=True)
    responses = trajectory.get("responses") or []
    recorded_model = (original_response or (responses[-1] if responses else {})).get("model")
    model = args.model or os.environ.get("OPENAI_MODEL") or recorded_model
    if not model:
        parser.error("Specify --model or set OPENAI_MODEL; no recorded model was found")
    if [m.get("role") for m in before[:2]] != ["system", "user"]:
        parser.error("Expected original system and task messages first")
    assistant_indices = [i for i, m in enumerate(before) if m.get("role") == "assistant"]
    if args.keep_recent < 1 or len(assistant_indices) <= args.keep_recent:
        parser.error("Need at least one old step and --keep-recent >= 1")
    split = assistant_indices[-args.keep_recent]
    old_messages = before[2:split]
    recent_messages = before[split:]
    serialized_history = serialize_conversation(old_messages)
    compaction_prompt = [
        {
            "role": "system",
            "content": COMPACTION_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Summarize the old history below using the original task as context.\n\n"
            + "[Original task]: " + content_text(before[1]["content"])
            + "\n\n[History to summarize]\n" + serialized_history,
        },
    ]
    request = {
        "model": model,
        "messages": compaction_prompt,
        "reasoning_effort": "medium",
        "max_completion_tokens": args.max_tokens,
    }
    print("\n" + "=" * 72)
    print(f"SERIALIZED HISTORY ({len(old_messages)} messages; {len(serialized_history):,} characters)")
    print("=" * 72)
    print(serialized_history or "(empty)")
    print("=" * 72 + "\n", flush=True)
    print(f"Calling {request['model']} with old history wrapped as user-message data...", flush=True)
    print("No tools or tool_choice parameter is supplied.", flush=True)
    with OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
        max_retries=0,
        timeout=120,
    ) as client:
        response = client.chat.completions.create(**request).model_dump(mode="json")

    choice = response["choices"][0]
    message = choice["message"]
    summary = (message.get("content") or "").strip()
    candidate = before[:2] + [
        {"role": "assistant", "content": "[COMPACTED CONTEXT]\n" + summary},
    ] + recent_messages
    checks = {
        "Nonempty text summary": bool(summary),
        "No tool calls": not message.get("tool_calls") and not message.get("function_call"),
        "Completed normally (not truncated)": choice["finish_reason"] == "stop",
        "Prompt got shorter": rough_message_tokens(candidate) < rough_message_tokens(before),
        "System/task unchanged": candidate[:2] == before[:2],
        "Recent steps unchanged": candidate[-len(recent_messages):] == recent_messages,
    }
    valid = all(checks.values())
    # Retain the original history on any validation failure.
    after = candidate if valid else before
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "request": request,
        "response": response,
        "original_response": original_response,
        "prompt_before": before,
        "candidate_prompt": candidate,
        "prompt_after": after,
        "checks": checks,
    }, indent=2, ensure_ascii=False) + "\n")

    print("\n" + "=" * 72)
    print(f"COMPACTED SUMMARY ({len(summary.split()):,} words; {len(summary):,} characters)")
    print("=" * 72)
    print(summary or "(empty summary)")
    print("=" * 72 + "\n")
    print(f"finish_reason: {choice['finish_reason']}")
    if message.get("tool_calls"):
        print("Unexpected tool calls:")
        print(json.dumps(message["tool_calls"], indent=2, ensure_ascii=False))
    print(f"Full request and response: {args.output}")

    print(f"Estimated tokens: {rough_message_tokens(before)} -> {rough_message_tokens(candidate)}")
    for label, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'}: {label}")
    if not valid:
        print("Validation failed; original history retained.")
    print("These checks validate structure and size, not factual summary quality.")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
