"""Inspect adapter that produces paired baseline/candidate evidence for Skill evolution.

The adapter executes the real SwarmX runtime through `swarmx eval-run` with the
same request-scoped Skill delivery used by production, scoring each hidden
holdout case deterministically. It never writes the active revision and never
decides promotion: it only emits evidence JSON that `swarmx evolution evaluate
--evidence` records in Core.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shlex
import signal
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
DEFAULT_CONFIG = str(ROOT / "fixtures" / "echo.swarm.json")

EVIDENCE_SCORER_FINGERPRINT = "swarmx.inspect.deterministic.v1"
EVIDENCE_RUNTIME_FINGERPRINT = "swarmx.inspect.cli-eval-run.v1"


@task
def skill_paired_eval(
    holdout: str = "",
    baseline_content: str = "",
    candidate_content: str = "",
    baseline_revision: str = "",
    candidate_revision: str = "",
    baseline_digest: str = "",
    candidate_digest: str = "",
    skill_id: str = "skill",
    variant_id: str = "skill:default",
    target_agent: str = "",
    target_model_fingerprint: str = "",
    target_agent_name: str = "",
    config: str = DEFAULT_CONFIG,
    command: str = DEFAULT_COMMAND,
    timeout: float = 300.0,
    seed: int = 0,
    evidence_output: str = "",
) -> Task:
    """Paired hidden-holdout evidence for one Skill candidate.

    Each holdout case runs through the same real SwarmX eval-run path twice:
    once with the baseline `prompt_fragment` delivery and once with the
    candidate delivery, in the seeded-randomized order that is recorded. The
    scorer is a deterministic target match; an optional blind LLM judge is a
    later phase. The evidence binds the holdout digest, sample count, target
    agent, and a config-derived runtime fingerprint; Core re-verifies them
    before any promotion.
    """
    if not holdout or not baseline_content or not candidate_content:
        raise ValueError(
            "skill_paired_eval requires holdout, baseline_content, and candidate_content paths"
        )
    if not target_agent or not target_model_fingerprint:
        raise ValueError(
            "skill_paired_eval requires target_agent and target_model_fingerprint matching the optimization request"
        )
    config_path = Path(config)
    config_text = config_path.read_bytes()
    assert_eval_safe_config(config_text)
    holdout_bytes = Path(holdout).read_bytes()
    holdout_digest = "sha256:" + hashlib.sha256(holdout_bytes).hexdigest()
    holdout_case_count = sum(
        1 for line in holdout_bytes.decode("utf-8").splitlines() if line.strip()
    )
    runtime_fingerprint = (
        "swarmx.inspect.config:" + hashlib.sha256(config_text).hexdigest()
    )
    agent_name = resolve_target_agent_name(config_text, target_agent_name)
    return Task(
        dataset=json_dataset(holdout, sample_to_evidence_case),
        solver=paired_solver(
            command=command_parts(command),
            config=config,
            target_agent_name=agent_name,
            baseline=PairedArtifact(
                baseline_content, baseline_revision, baseline_digest
            ),
            candidate=PairedArtifact(
                candidate_content, candidate_revision, candidate_digest
            ),
            skill_id=skill_id,
            variant_id=variant_id,
            timeout=timeout,
            seed=seed,
            evidence_output=evidence_output,
            holdout_digest=holdout_digest,
            holdout_case_count=holdout_case_count,
            target_agent=target_agent,
            target_model_fingerprint=target_model_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        ),
        scorer=paired_scorer(),
    )


def resolve_target_agent_name(config_bytes: bytes, requested: str) -> str:
    config = json.loads(config_bytes.decode("utf-8"))
    nodes = config.get("nodes", {})
    agents = [
        node.get("agent", {}).get("name")
        for node in nodes.values()
        if isinstance(node, dict) and node.get("kind") == "agent"
    ]
    if requested:
        if requested not in agents:
            raise ValueError(
                f"target_agent_name {requested!r} is not an agent in the eval config"
            )
        return requested
    if len(agents) == 1:
        return agents[0]
    raise ValueError("eval config has multiple agents; target_agent_name is required")


def assert_eval_safe_config(config_bytes: bytes) -> None:
    """Reject side-effect surfaces so evidence never runs real external tools.

    Mirrors Core's assertEvalSafeSwarmConfig for the Inspect side.
    """
    config = json.loads(config_bytes.decode("utf-8"))
    if not isinstance(config, dict):
        raise ValueError("eval config must be a JSON object")
    if config.get("queen"):
        raise ValueError("eval config must not define a queen agent")
    if config.get("mcpServers"):
        raise ValueError("eval config must not configure MCP servers")
    if config.get("hooks"):
        raise ValueError("eval config must not configure hooks")
    nodes = config.get("nodes", {})
    if not isinstance(nodes, dict):
        raise ValueError("eval config nodes must be an object")
    for name, node in nodes.items():
        if not isinstance(node, dict) or node.get("kind") != "agent":
            raise ValueError(f"eval config node {name!r} must be an agent node")
        agent = node.get("agent") or {}
        backend = agent.get("backend", {}).get("type", "swarmx")
        if backend not in {"swarmx", "echo"}:
            raise ValueError(
                f"eval config agent {name!r} uses backend {backend!r}; only direct native execution is supported"
            )
        if agent.get("mcpServers"):
            raise ValueError(
                f"eval config agent {name!r} must not configure MCP servers"
            )


class PairedArtifact:
    def __init__(self, content_path: str, revision: str, digest: str) -> None:
        self.content_path = content_path
        self.revision = revision
        self.digest = digest


def sample_to_evidence_case(record: dict[str, Any]) -> Sample:
    metadata = {
        key: value
        for key, value in record.items()
        if key not in {"id", "caseId", "input", "target"}
    }
    case_id = record.get("caseId") or record.get("id")
    return Sample(
        id=case_id,
        input=record["input"],
        target=record.get("target", ""),
        metadata=metadata,
    )


@solver
def paired_solver(
    command: list[str],
    config: str,
    target_agent_name: str,
    baseline: PairedArtifact,
    candidate: PairedArtifact,
    skill_id: str,
    variant_id: str,
    timeout: float,
    seed: int,
    evidence_output: str = "",
    holdout_digest: str = "",
    holdout_case_count: int = 0,
    target_agent: str = "",
    target_model_fingerprint: str = "",
    runtime_fingerprint: str = "",
) -> Solver:
    collected: list[dict[str, Any]] = []

    def write_evidence() -> None:
        if not evidence_output:
            return
        build_evidence_file(
            samples=list(collected),
            baseline_revision=baseline.revision,
            candidate_revision=candidate.revision,
            holdout_digest=holdout_digest,
            holdout_case_count=holdout_case_count,
            seed=seed,
            output_path=evidence_output,
            target_agent=target_agent,
            target_model_fingerprint=target_model_fingerprint,
            runtime_fingerprint=runtime_fingerprint,
        )

    async def solve(state: TaskState, _generate: Generate) -> TaskState:
        case_id = str(
            state.sample_id if state.sample_id is not None else state.input_text
        )
        candidate_ran_first = seeded_bit(seed, case_id)
        expected = state.target.text.strip() if state.target else ""
        expected_contains = string_or_none(state.metadata.get("expectedOutputContains"))
        safety_flag = string_or_none(state.metadata.get("safetyFlag"))

        # Actually follow the seeded order so caching, rate limits, and
        # stateful models cannot systematically favor one side.
        if candidate_ran_first:
            candidate_result = await run_eval_run(
                command=command,
                config=config,
                case=case_id,
                input_text=state.input_text,
                artifact=candidate,
                skill_id=skill_id,
                variant_id=variant_id,
                target_agent_name=target_agent_name,
                timeout=timeout,
            )
            baseline_result = await run_eval_run(
                command=command,
                config=config,
                case=case_id,
                input_text=state.input_text,
                artifact=baseline,
                skill_id=skill_id,
                variant_id=variant_id,
                target_agent_name=target_agent_name,
                timeout=timeout,
            )
        else:
            baseline_result = await run_eval_run(
                command=command,
                config=config,
                case=case_id,
                input_text=state.input_text,
                artifact=baseline,
                skill_id=skill_id,
                variant_id=variant_id,
                target_agent_name=target_agent_name,
                timeout=timeout,
            )
            candidate_result = await run_eval_run(
                command=command,
                config=config,
                case=case_id,
                input_text=state.input_text,
                artifact=candidate,
                skill_id=skill_id,
                variant_id=variant_id,
                target_agent_name=target_agent_name,
                timeout=timeout,
            )

        baseline_run = score_run(
            baseline_result, expected, expected_contains, safety_flag
        )
        candidate_run = score_run(
            candidate_result, expected, expected_contains, safety_flag
        )
        sample = {
            "caseId": case_id,
            "candidateRanFirst": candidate_ran_first,
            "baseline": baseline_run,
            "candidate": candidate_run,
        }
        state.metadata = {
            **state.metadata,
            "swarmx_paired": {
                "caseId": case_id,
                "candidateRanFirst": candidate_ran_first,
                "baseline": {
                    "revisionId": baseline.revision,
                    "digest": baseline.digest,
                    "output": baseline_result.get("output", ""),
                    "error": baseline_result.get("error"),
                    "latencyMs": baseline_result.get("latencyMs", 0),
                },
                "candidate": {
                    "revisionId": candidate.revision,
                    "digest": candidate.digest,
                    "output": candidate_result.get("output", ""),
                    "error": candidate_result.get("error"),
                    "latencyMs": candidate_result.get("latencyMs", 0),
                },
                "sample": sample,
            },
        }
        collected.append(sample)
        write_evidence()
        state.output = ModelOutput.from_content(
            model="swarmx",
            content=str(candidate_result.get("output", "")),
            error=string_or_none(candidate_result.get("error")),
        )
        return state

    return solve


def score_run(
    result: dict[str, Any],
    expected: str,
    expected_contains: str | None,
    safety_flag: str | None,
) -> dict[str, Any]:
    """Deterministic per-side scoring identical in spirit to Core's scorer."""
    output = str(result.get("output", ""))
    error = string_or_none(result.get("error"))
    failed = error is not None
    passed = False
    if not failed:
        if expected:
            passed = expected in output
        elif expected_contains:
            passed = expected_contains in output
        else:
            passed = bool(output.strip())
    safety_passed = not failed and (not safety_flag or safety_flag not in output)
    metrics = result.get("metrics")
    context_tokens = 0
    if isinstance(metrics, dict):
        value = metrics.get("contextTokens")
        if isinstance(value, int) and value >= 0:
            context_tokens = value
    return {
        "passed": passed,
        "safetyPassed": safety_passed,
        "contextTokens": context_tokens,
        "latencyMs": int(result.get("latencyMs", 0)),
        "failed": failed,
    }


@scorer(metrics=[accuracy(), stderr()])
def paired_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        paired = state.metadata.get("swarmx_paired")
        if not isinstance(paired, dict):
            return Score(value=INCORRECT, explanation="Missing swarmx_paired metadata")
        sample = paired.get("sample")
        if not isinstance(sample, dict):
            return Score(value=INCORRECT, explanation="Missing paired sample")
        candidate_run = sample.get("candidate")
        baseline_run = sample.get("baseline")
        if not isinstance(candidate_run, dict):
            return Score(value=INCORRECT, explanation="Missing candidate run")
        if candidate_run.get("failed"):
            return Score(
                value=INCORRECT,
                explanation="Candidate execution failed on the holdout case",
            )
        baseline_passed = isinstance(baseline_run, dict) and bool(
            baseline_run.get("passed")
        )
        candidate_passed = bool(candidate_run.get("passed")) and bool(
            candidate_run.get("safetyPassed")
        )
        if not candidate_run.get("safetyPassed"):
            return Score(
                value=INCORRECT,
                explanation="Candidate failed the holdout case or its safety check",
            )
        if not baseline_passed and candidate_passed:
            return Score(
                value=CORRECT, explanation="Candidate improved on the holdout case"
            )
        if candidate_passed:
            return Score(
                value=CORRECT, explanation="Candidate matched the holdout target"
            )
        return Score(
            value=INCORRECT, explanation="Candidate did not match the holdout target"
        )

    return score


async def run_eval_run(
    command: list[str],
    config: str,
    case: str,
    input_text: str,
    artifact: PairedArtifact,
    skill_id: str,
    variant_id: str,
    target_agent_name: str,
    timeout: float,
) -> dict[str, Any]:
    delivery = json.dumps(
        {
            "skillId": skill_id,
            "variantId": variant_id,
            "revisionId": artifact.revision,
            "contentDigest": artifact.digest,
            "mode": "prompt_fragment",
        },
        separators=(",", ":"),
    )
    args = [
        *command,
        "eval-run",
        "--config",
        config,
        "--input-json",
        json.dumps(
            {"messages": [{"role": "user", "content": input_text}]},
            separators=(",", ":"),
        ),
        "--skill-delivery",
        delivery,
        "--skill-content-path",
        artifact.content_path,
        "--skill-delivery-agent",
        target_agent_name,
    ]
    started = asyncio.get_event_loop().time()
    try:
        proc = await asyncio.create_subprocess_exec(
            *args,
            start_new_session=True,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )
        except asyncio.TimeoutError:
            await kill_process_tree(proc)
            raise
    except asyncio.TimeoutError:
        return {
            "output": "",
            "error": f"swarmx eval-run timed out for case {case}",
            "latencyMs": int((asyncio.get_event_loop().time() - started) * 1000),
        }
    except OSError as exc:
        return {"output": "", "error": str(exc), "latencyMs": 0}
    latency_ms = int((asyncio.get_event_loop().time() - started) * 1000)
    stdout_text = stdout.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(stdout_text)
        if isinstance(parsed, dict):
            return {
                "output": str(parsed.get("output", "")),
                "error": string_or_none(parsed.get("error")),
                "latencyMs": latency_ms,
                "metrics": parsed.get("metrics"),
            }
        return {
            "output": "",
            "error": "eval-run did not return a JSON object",
            "latencyMs": latency_ms,
        }
    except json.JSONDecodeError as exc:
        return {
            "output": "",
            "error": f"failed to parse eval-run JSON: {exc}",
            "latencyMs": latency_ms,
        }


def build_evidence_file(
    samples: list[dict[str, Any]],
    baseline_revision: str,
    candidate_revision: str,
    holdout_digest: str,
    holdout_case_count: int,
    seed: int,
    output_path: str,
    target_agent: str = "",
    target_model_fingerprint: str = "",
    runtime_fingerprint: str = EVIDENCE_RUNTIME_FINGERPRINT,
) -> None:
    """Serialize paired evidence into the Core evidence JSON contract."""
    evidence = {
        "evaluatorId": "inspect.skill_paired_eval",
        "scorerFingerprint": EVIDENCE_SCORER_FINGERPRINT,
        "runtimeFingerprint": runtime_fingerprint,
        "seed": seed,
        "holdoutContentDigest": holdout_digest,
        "holdoutCaseCount": holdout_case_count,
        "baselineRevisionId": baseline_revision,
        "candidateRevisionId": candidate_revision,
        "samples": samples,
    }
    evidence["targetAgentId"] = target_agent
    if target_model_fingerprint:
        evidence["targetModelFingerprint"] = target_model_fingerprint
    Path(output_path).write_text(json.dumps(evidence, indent=2), "utf-8")


def evidence_from_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert inspect evaluation results into Core SkillEvaluationSample records."""
    samples: list[dict[str, Any]] = []
    for record in records:
        paired = record.get("metadata", {}).get("swarmx_paired")
        if not isinstance(paired, dict):
            continue
        sample = paired.get("sample")
        if not isinstance(sample, dict):
            continue
        samples.append(
            {
                "caseId": sample["caseId"],
                "candidateRanFirst": bool(sample["candidateRanFirst"]),
                "baseline": sample["baseline"],
                "candidate": sample["candidate"],
            }
        )
    return samples


async def kill_process_tree(proc: asyncio.subprocess.Process) -> None:
    """Kill the child and its whole process group on POSIX."""
    if proc.returncode is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.kill()
    except ProcessLookupError:
        pass
    try:
        await proc.wait()
    except (ProcessLookupError, ChildProcessError):
        pass


def seeded_bit(seed: int, case_id: str) -> bool:
    h = (seed ^ hash_stable(case_id)) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) & 0xFFFFFFFF
    return (h & 1) == 1


def hash_stable(value: str) -> int:
    result = 0
    for char in value:
        result = (result * 31 + ord(char)) & 0xFFFFFFFF
    return result


def command_parts(command: str | Sequence[str]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command)
    return list(command)


def string_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) else None
