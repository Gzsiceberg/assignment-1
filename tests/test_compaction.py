"""Offline checks for compaction serialization and safe history replacement."""

from copy import deepcopy
from types import SimpleNamespace

from openai.types.chat import ChatCompletion
import pytest

from assignment.agent.base import Agent, serialize_conversation


def completion(content, finish_reason="stop", tool_calls=None):
    return ChatCompletion.model_validate({
        "id": "test", "object": "chat.completion", "created": 0, "model": "test",
        "choices": [{
            "index": 0, "finish_reason": finish_reason,
            "message": {"role": "assistant", "content": content, "tool_calls": tool_calls},
        }],
    })


def test_serializer_omits_metadata_and_truncates_results():
    text = serialize_conversation([
        {
            "role": "assistant", "content": "Inspect file.", "reasoning_content": "SECRET",
            "tool_calls": [{"id": "ignored-id", "function": {
                "name": "execute", "arguments": '{"command": "cat f.txt"}',
            }}],
        },
        {"role": "tool", "content": "x" * 2100},
        {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "SECRET"},
            {"type": "text", "text": "Visible"},
        ]},
    ])
    assert '[Assistant tool calls]: execute(command="cat f.txt")' in text
    assert "[... 100 characters truncated]" in text
    assert "SECRET" not in text and "ignored-id" not in text
    assert "[Assistant]: Visible" in text


@pytest.fixture
def agent():
    # No client construction or environment: compact_context only uses these fields.
    agent = Agent.__new__(Agent)
    agent.system_prompt = "Standing instructions"
    agent.task_prompt = "Original task"
    agent.model = "test"
    agent.compaction_keep_recent_steps = 2
    agent.compaction_max_tokens = 1200
    agent.assistant_messages = []
    for index in range(3):
        agent.assistant_messages.extend([
            {"role": "assistant", "content": f"step-{index}", "tool_calls": [{
                "id": str(index), "type": "function",
                "function": {"name": "execute", "arguments": '{"command": "ls"}'},
            }]},
            {"role": "tool", "tool_call_id": str(index), "content": "x" * 4000},
        ])
    return agent


def attach_response(agent, response):
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return response

    agent.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return calls


def test_compaction_preserves_configured_recent_steps(agent):
    before = deepcopy(agent.build_prompt())
    response = completion("Inspected files; continue investigating.")
    calls = attach_response(agent, response)
    prompt, returned = agent.compact_context()
    after = agent.build_prompt()
    assert after[:2] == before[:2]
    assert after[-4:] == before[-4:]
    assert "step-0" in prompt[1]["content"]
    assert "step-1" not in prompt[1]["content"]
    assert "[COMPACTED CONTEXT]" in after[2]["content"]
    assert returned == response.model_dump(mode="json")
    assert "tools" not in calls[0]


@pytest.mark.parametrize("response", [
    completion(""),
    completion("Incomplete summary", finish_reason="length"),
    completion("x" * 20000),
    completion("", finish_reason="tool_calls", tool_calls=[{
        "id": "unexpected", "type": "function",
        "function": {"name": "execute", "arguments": "{}"},
    }]),
])
def test_invalid_compaction_leaves_history_unchanged(agent, response):
    before = deepcopy(agent.build_prompt())
    attach_response(agent, response)
    with pytest.raises(RuntimeError):
        agent.compact_context()
    assert agent.build_prompt() == before
