"""GEPA skill optimization runner for the swarmx.evolve_skill operation.

The metric returns a scalar score plus actionable textual feedback and a
deterministic validator result. Reflection/proposal either goes through the
host capability gateway (`proposer: gateway`) or uses the offline
deterministic proposer (`proposer: deterministic`, test/no-network mode).
There is no silent optimizer fallback: failures surface in the report.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any, Callable

import dspy

from .capability_lm import CapabilityLm, DeterministicLm
from .skill_program import build_skill_program, export_skill_markdown, skill_examples
from .errors import RsiCancelledError

ProgressCallback = Callable[[str, float], None]
CancelCheck = Callable[[], bool]
CheckpointCallback = Callable[[str, dict[str, Any]], None]


def canonical_config_digest(request: dict[str, Any]) -> str:
    """Canonical optimizer config derived from the granted request."""
    optimizer = request["optimizer"]
    budget = request.get("budget", {})
    config = {
        "schemaVersion": 1,
        "optimizerId": "dspy.gepa.v1",
        "seed": int(optimizer["seed"]),
        "proposer": request.get("proposer", "gateway"),
        "budget": {
            "maxModelCalls": budget.get("maxModelCalls"),
            "maxTokens": budget.get("maxTokens"),
            "maxWallTimeMs": budget.get("maxWallTimeMs"),
        },
    }
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def run_gepa(
    request: dict[str, Any],
    capability_client: Any,
    cancel_check: CancelCheck,
    progress: ProgressCallback,
    checkpoint: Callable[[str, dict[str, Any]], None] | None = None,
) -> tuple[dict[str, Any], str]:
    """Run GEPA over the baseline Skill and return (report, candidate Markdown)."""
    start_wall = time.monotonic()
    if cancel_check():
        raise RsiCancelledError("Skill optimization cancelled before start.")

    def stage(name: str, state: dict[str, Any]) -> None:
        if cancel_check():
            raise RsiCancelledError(f"Skill optimization cancelled during {name}.")
        if checkpoint is not None:
            checkpoint(name, state)

    stage("reading_artifacts", {"phase": "reading_artifacts"})
    baseline = read_artifact(capability_client, request["baselineContentRef"])
    train_text = read_artifact(capability_client, request["trainDataset"]["contentRef"])
    dev_text = read_artifact(capability_client, request["devDataset"]["contentRef"])
    train_records = parse_records(train_text)
    dev_records = parse_records(dev_text)
    if not train_records:
        raise ValueError("The granted train dataset contains no usable records.")
    trainset = skill_examples(train_records)
    devset = skill_examples(dev_records)
    seed = int(request["optimizer"]["seed"])
    proposer = request.get("proposer", "gateway")
    errors: list[str] = []
    max_model_calls = request.get("budget", {}).get("maxModelCalls")
    if not isinstance(max_model_calls, int) or max_model_calls <= 0:
        raise ValueError(
            "dspy.gepa.v1 requires a positive maxModelCalls budget; none was granted."
        )
    max_metric_calls = max_model_calls

    if proposer == "gateway":
        reflection_lm = CapabilityLm(
            capability_client, model=request.get("targetModelFingerprint")
        )
        instruction_proposer = None
    elif proposer == "deterministic":
        reflection_lm = None
        instruction_proposer = DeterministicInstructionProposer()
    else:
        raise ValueError(f"Unsupported GEPA proposer mode: {proposer}")

    stage("compiling", {"phase": "compiling"})
    progress("Compiling GEPA optimizer.", 0.1)
    program_lm: Any = None
    try:
        program_lm = deterministic_program_lm(proposer, capability_client)
        with dspy.context(lm=program_lm):
            program = build_skill_program(baseline)
            metric = skill_gepa_metric
            gepa = dspy.GEPA(
                metric=metric,
                max_metric_calls=max_metric_calls,
                reflection_minibatch_size=2,
                reflection_lm=reflection_lm,
                instruction_proposer=instruction_proposer,
                skip_perfect_score=True,
                candidate_selection_strategy="pareto",
                use_merge=False,
                num_threads=1,
                failure_score=0.0,
                perfect_score=1.0,
                seed=seed,
                track_stats=True,
            )
            stage("optimizing", {"phase": "optimizing"})
            progress("Running GEPA compile.", 0.3)
            optimized = gepa.compile(program, trainset=trainset, valset=devset)
            stage("exporting", {"phase": "exporting"})
            candidate_markdown = export_skill_markdown(optimized)
    except RsiCancelledError:
        raise
    except Exception as error:  # noqa: BLE001 - surfaced explicitly in the report
        errors.append(f"{type(error).__name__}: {error}")
        raise

    wall_ms = int((time.monotonic() - start_wall) * 1_000)
    calls, tokens = collect_stats(program_lm, reflection_lm, proposer)
    report = {
        "optimizerId": "dspy.gepa.v1",
        "optimizerVersion": "dspy.gepa.v1",
        "actualOptimizer": "dspy.gepa.v1",
        "proposer": proposer,
        "seed": seed,
        "maxMetricCalls": max_metric_calls,
        "modelCalls": calls,
        "tokens": tokens,
        "wallTimeMs": wall_ms,
        "errors": errors,
    }
    progress("GEPA compile finished.", 1.0)
    return report, candidate_markdown


def skill_gepa_metric(
    gold: dspy.Example,
    pred: Any,
    trace: Any = None,
    pred_name: str | None = None,
    pred_trace: Any = None,
    program_trace: Any = None,
) -> dict[str, Any]:
    """Scalar score plus actionable textual feedback plus a deterministic check.

    Deterministic validator: the predicted answer must equal the gold answer.
    The feedback names the expected answer so the proposer can act on it.
    Returns a `ScoreWithFeedback` so GEPA can attach the feedback text.
    """
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    expected = getattr(gold, "answer", "") or ""
    actual = getattr(pred, "answer", "") or ""
    passed = expected.strip() == actual.strip()
    feedback = (
        f"The answer is correct ({expected!r})."
        if passed
        else f"The answer must be exactly `{expected}`; the model produced {actual!r}."
    )
    return ScoreWithFeedback(score=1.0 if passed else 0.0, feedback=feedback)


class DeterministicInstructionProposer:
    """Offline proposer: appends the missing gold answer as a mandatory rule.

    Used only when the explicit `proposer: deterministic` config is granted; it
    never runs when a gateway LM is available. The expected answer is extracted
    from the metric feedback text, so the proposal is grounded in evidence.
    """

    def __call__(
        self,
        candidate: dict[str, str],
        reflective_dataset: dict[str, Any],
        components_to_update: list[str],
    ) -> dict[str, str]:
        expected = first_expected_answer(reflective_dataset)
        updated = dict(candidate)
        for component in components_to_update:
            instruction = updated.get(component, "")
            if expected and expected not in instruction:
                updated[component] = (
                    f"{instruction}\n\nMandatory rule: the final answer must be exactly "
                    f"`{expected}` and nothing else."
                )
        return updated


def first_expected_answer(reflective_dataset: dict[str, Any]) -> str | None:
    match = re.search(
        r"exactly\s+`([A-Za-z0-9][A-Za-z0-9._-]*)`", str(reflective_dataset)
    )
    return match.group(1) if match else None


def deterministic_program_lm(proposer: str, capability_client: Any) -> Any:
    if proposer == "gateway":
        return CapabilityLm(capability_client)
    return DeterministicLm()


def collect_stats(
    program_lm: Any, reflection_lm: Any, proposer: str
) -> tuple[int, int]:
    calls = 0
    tokens = 0
    for lm in (program_lm, reflection_lm):
        if lm is None or not hasattr(lm, "stats"):
            continue
        stats = lm.stats()
        calls += int(stats.get("calls", 0))
        tokens += int(stats.get("tokens", 0))
    return calls, tokens


def read_artifact(capability_client: Any, ref: str) -> str:
    outcome = capability_client.call("skill_evolution", "read_artifact", {"ref": ref})
    if outcome.get("status") != "succeeded":
        raise ValueError(f"Granted artifact {ref} could not be read.")
    value = outcome.get("value") or {}
    content = value.get("content")
    if not isinstance(content, str):
        raise ValueError(f"Granted artifact {ref} has no text content.")
    return content


def parse_records(text: str) -> list[dict[str, Any]]:
    stripped = text.strip()
    if stripped:
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [record for record in parsed if isinstance(record, dict)]
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    records: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records
