import asyncio
import json
import shlex
from pathlib import Path
from typing import Any, Sequence

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, json_dataset
from inspect_ai.model import ModelOutput
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

ROOT = Path(__file__).resolve().parent
DEFAULT_COMMAND = "node packages/cli/dist/cli.js"
DEFAULT_DATASET = str(ROOT / "datasets" / "smoke.jsonl")
DEFAULT_CONFIG = str(ROOT / "fixtures" / "missing-root.swarm.json")


@solver
def swarmx_solver(
    config: str = DEFAULT_CONFIG,
    command: str | Sequence[str] = DEFAULT_COMMAND,
    timeout: float = 300.0,
    cwd: str | None = None,
) -> Solver:
    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        result, stderr_text, exit_code = await run_swarmx_eval(
            command=command_parts(command),
            config=sample_config(state.metadata, config),
            input_json=sample_arguments(state),
            timeout=timeout,
            cwd=cwd,
        )
        state.metadata = {
            **state.metadata,
            "swarmx_result": result,
            "swarmx_stderr": stderr_text,
            "swarmx_exit_code": exit_code,
        }
        state.output = ModelOutput.from_content(
            model="swarmx",
            content=str(result.get("output", "")),
            error=string_or_none(result.get("error")),
        )
        state.messages.append(state.output.message)
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def swarmx_contract_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        result = state.metadata.get("swarmx_result")
        if not isinstance(result, dict):
            return Score(value=INCORRECT, explanation="Missing swarmx_result metadata")

        failures = contract_failures(result, state.metadata, target)
        return Score(
            value=CORRECT if not failures else INCORRECT,
            answer=str(result.get("output") or result.get("error") or ""),
            explanation="\n".join(failures)
            if failures
            else "SwarmX eval contract satisfied",
        )

    return score


@task
def swarmx_smoke(
    config: str = DEFAULT_CONFIG,
    dataset: str = DEFAULT_DATASET,
    command: str = DEFAULT_COMMAND,
    timeout: float = 300.0,
) -> Task:
    return Task(
        dataset=json_dataset(dataset, sample_to_eval),
        solver=swarmx_solver(config=config, command=command, timeout=timeout),
        scorer=swarmx_contract_scorer(),
    )


def sample_to_eval(record: dict[str, Any]) -> Sample:
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"id", "input", "target"}
    }
    return Sample(
        id=record.get("id"),
        input=record["input"],
        target=record.get("target", ""),
        metadata=metadata,
    )


def sample_arguments(state: TaskState) -> dict[str, Any]:
    arguments = state.metadata.get("arguments")
    if isinstance(arguments, dict):
        return arguments
    return {"messages": [{"role": "user", "content": state.input_text}]}


def sample_config(metadata: dict[str, Any], default_config: str) -> str:
    config = metadata.get("config")
    return config if isinstance(config, str) else default_config


async def run_swarmx_eval(
    command: list[str],
    config: str,
    input_json: dict[str, Any],
    timeout: float,
    cwd: str | None,
) -> tuple[dict[str, Any], str, int | None]:
    args = [
        *command,
        "eval-run",
        "--config",
        config,
        "--input-json",
        json.dumps(input_json, separators=(",", ":")),
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        return error_result(f"swarmx eval-run timed out after {timeout:g}s"), "", None
    except OSError as exc:
        return error_result(str(exc)), "", None

    stdout_text = stdout.decode("utf-8", errors="replace")
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(stdout_text)
        if isinstance(parsed, dict):
            return parsed, stderr_text, proc.returncode
        return (
            error_result("swarmx eval-run did not return a JSON object"),
            stderr_text,
            proc.returncode,
        )
    except json.JSONDecodeError as exc:
        message = f"failed to parse swarmx eval-run JSON: {exc}"
        return error_result(message), stdout_text + stderr_text, proc.returncode


def contract_failures(
    result: dict[str, Any], metadata: dict[str, Any], target: Target
) -> list[str]:
    failures: list[str] = []
    output = str(result.get("output", ""))
    error = string_or_none(result.get("error"))
    trace = result.get("trace")
    metrics = result.get("metrics")

    if not isinstance(result.get("messages"), list):
        failures.append("messages is not a list")
    if not isinstance(trace, list):
        failures.append("trace is not a list")
    if not isinstance(metrics, dict):
        failures.append("metrics is not an object")

    expect_error = string_or_none(metadata.get("expect_error_contains"))
    if expect_error:
        if not error or expect_error not in error:
            failures.append(
                f"expected error containing {expect_error!r}, got {error!r}"
            )
    elif error:
        failures.append(f"unexpected eval error: {error}")

    target_text = target.text.strip()
    if target_text and target_text not in output:
        failures.append(f"target text {target_text!r} not found in output")

    expect_output = string_or_none(metadata.get("expect_output_contains"))
    if expect_output and expect_output not in output:
        failures.append(f"expected output containing {expect_output!r}")

    expected_nodes = metadata.get("expected_nodes")
    if isinstance(expected_nodes, list) and isinstance(trace, list):
        actual_nodes = [event.get("node") for event in trace if isinstance(event, dict)]
        if actual_nodes != expected_nodes:
            failures.append(f"expected nodes {expected_nodes!r}, got {actual_nodes!r}")

    expected_steps = metadata.get("expected_steps")
    if isinstance(expected_steps, int) and isinstance(metrics, dict):
        if metrics.get("steps") != expected_steps:
            failures.append(
                f"expected {expected_steps} steps, got {metrics.get('steps')!r}"
            )

    max_steps = metadata.get("max_steps")
    if isinstance(max_steps, int) and isinstance(metrics, dict):
        steps = metrics.get("steps")
        if isinstance(steps, int) and steps > max_steps:
            failures.append(f"expected at most {max_steps} steps, got {steps}")

    expected_tool_calls = metadata.get("expected_tool_calls")
    if isinstance(expected_tool_calls, int) and isinstance(metrics, dict):
        tool_calls = metrics.get("toolCalls")
        if tool_calls != expected_tool_calls:
            failures.append(
                f"expected {expected_tool_calls} tool calls, got {tool_calls!r}"
            )

    return failures


def command_parts(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return list(command)


def error_result(error: str) -> dict[str, Any]:
    return {
        "output": "",
        "messages": [],
        "trace": [],
        "error": error,
        "metrics": {
            "steps": 0,
            "messages": 0,
            "toolCalls": 0,
            "toolResults": 0,
        },
    }


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
