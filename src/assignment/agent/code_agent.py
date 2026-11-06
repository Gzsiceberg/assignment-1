"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations

import json
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, ValidationError

from assignment.agent.base import (
    DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
    DEFAULT_COMPACTION_MAX_TOKENS,
    Agent,
    format_tool_output,
)
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL, INVOKE_SKILL_TOOL
from assignment.env import Environment

NonBlankString = Annotated[str, StringConstraints(pattern=r"\S")]


class ToolArguments(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class ExecuteArguments(ToolArguments):
    command: NonBlankString | Annotated[list[str], Field(min_length=1)]
    shell: bool | None = True
    cwd: str | None = None
    timeout: float | None = None
    env: dict[str, str] | None = None


class SendMessageArguments(ToolArguments):
    summary: NonBlankString


class InvokeSkillArguments(ToolArguments):
    name: NonBlankString


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(
        self,
        task: str,
        environment: Environment,
        model: str | None = None,
        logs_save_path: str | None = None,
        step_limit: int = 100,
        skills_path: str | None = None,
        auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        super().__init__(
            environment=environment,
            model=model,
            logs_save_path=logs_save_path,
            step_limit=step_limit,
            skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.task = task
        self.submitted_patch = ""

        # TODO(Part 1.3): Make the `execute` and `send_message` tools available
        # to the agent.
        self.tools.append(EXECUTE_TOOL)
        self.tools.append(SEND_MESSAGE_TOOL)

        # TODO(1.1.b): Construct the system prompt and task_prompt. These
        # should be usable by the `Agent.build_prompt` method.
        system_info = {
            "machine": environment.machine,
            "release": environment.release,
            "system": environment.system,
            "version": environment.version,
        }
        self.system_prompt = f"""You are a code agent.
        <system_information>
        {json.dumps(system_info, indent=4)}
        </system_information>
        """
        self.task_prompt = f"Your task is to solve the following task: {self.task}"

        # TODO(1.4): If any skills are available to the agent, make their
        # descriptions/metadata available to the agent in the prompt.
        if self.skills:
            skills_markdown = ""
            for skill_name in self.skills:
                skill = self.skills[skill_name]
                skills_markdown += f"### {skill_name}\n"
                skills_markdown += f"{skill["metadata"]}\n\n"
            self.system_prompt += f"\n<skills>\n{skills_markdown}\n</skills>"

    @staticmethod
    def tool_error(tool_call_id, message: str) -> dict[str, str]:
        return {"role": "tool", "tool_call_id": tool_call_id, "content": message}

    def execute_command(self, tool_call_id, tool_call: dict[str, Any]) -> dict[str, str]:
        try:
            args = ExecuteArguments.model_validate_json(
                tool_call.get("function", {}).get("arguments", "{}")
            )
        except ValidationError as exc:
            return self.tool_error(tool_call_id, f"Invalid execute arguments: {exc}")
        result = self.env.execute(**args.model_dump())
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": format_tool_output(result),
        }

    def send_message(self, tool_call_id, tool_call: dict[str, Any]) -> dict[str, str]:
        try:
            args = SendMessageArguments.model_validate_json(
                tool_call.get("function", {}).get("arguments", "{}")
            )
        except ValidationError as exc:
            return self.tool_error(tool_call_id, f"Invalid send_message arguments: {exc}")
        self.finished = True
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": f"Message sent: {args.summary}",
        }

    def invoke_skills(self, tool_call_id, tool_call: dict[str, Any]) -> dict[str, str]:
        try:
            args = InvokeSkillArguments.model_validate_json(
                tool_call.get("function", {}).get("arguments", "{}")
            )
        except ValidationError as exc:
            return self.tool_error(tool_call_id, f"Invalid invoke_skill arguments: {exc}")
        skill_name = args.name
        if skill_name not in self.skills:
            return {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": f"Unknown skill: {skill_name}",
                    }
        skill = self.skills[skill_name]
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": skill["content"],
        }

    def execute_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, str]]:
        """Execute ``execute`` and ``send_message`` calls in the code sandbox."""

        # TODO(Part 1.3): Parse each call, execute recognized tools, and return
        # one message per call (there may be multiple tool calls in one agent
        # response!). Malformed JSON and unknown tools must become recoverable
        # observations relayed to the agent instead of exceptions.
        tool_results: list[dict[str, str]] = []
        execute_tool_name = EXECUTE_TOOL["function"]["name"]
        send_message_tool_name = SEND_MESSAGE_TOOL["function"]["name"]
        invoke_skills_tool_name = INVOKE_SKILL_TOOL["function"]["name"]
        valid_tool_names = [
            execute_tool_name,
            send_message_tool_name
        ]
        if self.skills:
            valid_tool_names.append(invoke_skills_tool_name)
        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            tool_call_id = tool_call.get("id")
            if not tool_call_id:
                tool_results.append(
                    {
                        "role": "tool",
                        "content": f"Tool call is missing an id: {tool_call}"
                    }
                )
                continue

            if tool_name not in valid_tool_names:
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": f"Unknown tool: {tool_name}",
                    }
                )
                continue

            if tool_name == execute_tool_name:
                tool_results.append(self.execute_command(tool_call_id, tool_call))
            elif tool_name == send_message_tool_name:
                tool_results.append(self.send_message(tool_call_id, tool_call))
            elif tool_name == invoke_skills_tool_name:
                tool_results.append(self.invoke_skills(tool_call_id, tool_call))

        return tool_results


