"""Reference Python backend for the SwarmX task worker protocol."""

import argparse
import asyncio
import hashlib
import json
import os
import platform
import re
import sys
import threading
import time
import uuid
from collections.abc import Mapping

PROTOCOL_VERSION = 1
WORKER_BACKEND_ID = "python"
WORKER_BACKEND_VERSION = "2"
MAX_LINE_BYTES = 1024 * 1024
ENVIRONMENT_DIGEST_PATTERN = re.compile(r"^sha256:[a-f0-9]{64}$")
REQUEST_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*$")

# The JSONL protocol owns fd 1. Library output (for example DSPy/GEPA progress
# bars) is redirected to stderr during optimizer runs so it can never corrupt
# the protocol stream, and heartbeats keep using this original writer.
_PROTOCOL_STDOUT = sys.stdout

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject = dict[str, JsonValue]


class ProtocolError(RuntimeError):
    """Raised when the trusted Core host sends an invalid control message."""


class CapabilityError(RuntimeError):
    """Raised when a granted capability call fails, times out, or is cancelled."""


class CapabilityTimeoutError(CapabilityError):
    pass


class CapabilityCancelledError(CapabilityError):
    pass


class CapabilityBudgetError(CapabilityError):
    pass


class WorkerRuntime:
    def __init__(self, environment_digest: str) -> None:
        self.environment_digest = environment_digest
        self.instance_id = f"python:{os.getpid()}:{uuid.uuid4().hex[:12]}"
        self.hello_message_id = self._message_id("hello")
        self.capabilities_received = False
        self.active_task: asyncio.Task[None] | None = None
        self.heartbeat_task: asyncio.Task[None] | None = None
        self.active_run: JsonObject | None = None
        self.cancel_event = asyncio.Event()
        self.cancel_mode = "cancel"
        self.cancel_reason = "Cancellation requested by the SwarmX control plane."
        self.sequence = 0
        self.heartbeat_interval_ms = 5_000
        self.capability_client: "CapabilityClient | None" = None
        self._last_artifact_id: str | None = None
        self._last_checkpoint_id: str | None = None

    async def run(self) -> int:
        self._write(
            {
                "protocolVersion": PROTOCOL_VERSION,
                "messageId": self.hello_message_id,
                "direction": "worker_to_host",
                "type": "hello",
                "worker": {
                    "instanceId": self.instance_id,
                    "backendId": WORKER_BACKEND_ID,
                    "backendVersion": WORKER_BACKEND_VERSION,
                    "language": "python",
                    "languageVersion": platform.python_version(),
                    "environmentDigest": self.environment_digest,
                },
                "supportedProtocolVersions": [PROTOCOL_VERSION],
                "operations": [
                    "swarmx.count",
                    "swarmx.echo",
                    "swarmx.fail",
                    "swarmx.needs_human",
                    "swarmx.evolve_skill",
                ],
                "features": [
                    "heartbeat",
                    "progress",
                    "checkpoint",
                    "artifact",
                    "needs_human",
                    "cancel",
                    "capability_gateway",
                ],
            }
        )

        while True:
            line = await asyncio.to_thread(
                sys.stdin.buffer.readline, MAX_LINE_BYTES + 1
            )
            if not line:
                await self._stop_active_task()
                return 0
            try:
                message = self._parse_control_line(line)
                await self._handle_control(message)
            except ProtocolError as error:
                print(
                    f"SwarmX worker protocol error: {error}",
                    file=sys.stderr,
                    flush=True,
                )
                await self._stop_active_task()
                return 2

    async def _handle_control(self, message: JsonObject) -> None:
        message_type = _required_string(message, "type")
        if message_type == "capabilities":
            if self.capabilities_received:
                raise ProtocolError("capabilities may be negotiated only once")
            if _required_string(message, "helloMessageId") != self.hello_message_id:
                raise ProtocolError("capabilities references an unknown hello message")
            if (
                _required_integer(message, "selectedProtocolVersion")
                != PROTOCOL_VERSION
            ):
                raise ProtocolError("the host selected an unsupported protocol version")
            limits = _required_mapping(message, "limits")
            self.heartbeat_interval_ms = _bounded_integer(
                limits.get("heartbeatIntervalMs"), "heartbeatIntervalMs", 1, 60_000
            )
            self.capabilities_received = True
            return

        if message_type == "start":
            if not self.capabilities_received:
                raise ProtocolError("start received before capabilities negotiation")
            if self.active_task and not self.active_task.done():
                raise ProtocolError("this worker instance already has an active run")
            if (
                _required_string(message, "environmentDigest")
                != self.environment_digest
            ):
                raise ProtocolError(
                    "start environment digest does not match this worker"
                )
            self.active_run = message
            self.cancel_event = asyncio.Event()
            self.cancel_mode = "cancel"
            self.cancel_reason = "Cancellation requested by the SwarmX control plane."
            self.sequence = 0
            grant_ids = message.get("capabilityGrantIds")
            if not isinstance(grant_ids, list):
                raise ProtocolError("start capabilityGrantIds must be an array")
            self.capability_client = CapabilityClient(
                self,
                grant_ids,
                self._capability_budget(message),
                _required_string(message, "runId"),
                asyncio.get_running_loop(),
                token_budget=self._request_token_budget(message),
            )
            self.active_task = asyncio.create_task(self._execute(message))
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.active_task.add_done_callback(self._task_finished)
            return

        if message_type == "cancel":
            if not self._matches_active_run(message):
                return
            self.cancel_mode = _required_string(message, "mode")
            self.cancel_reason = _required_string(message, "reason")
            self.cancel_event.set()
            return

        if message_type == "capability_result":
            if not self._matches_active_run(message):
                return
            client = getattr(self, "capability_client", None)
            if client is not None:
                client.deliver_result(message)
            return

        raise ProtocolError(f"unsupported host message type: {message_type}")

    async def _execute(self, start: JsonObject) -> None:
        operation = _required_mapping(start, "operation")
        operation_name = _required_string(operation, "name")
        operation_input = operation.get("input")
        if operation_name == "swarmx.echo":
            await self._execute_echo(operation_input)
        elif operation_name == "swarmx.count":
            await self._execute_count(start, operation_input)
        elif operation_name == "swarmx.fail":
            await self._execute_failure(operation_input)
        elif operation_name == "swarmx.needs_human":
            await self._execute_needs_human(start, operation_input)
        elif operation_name == "swarmx.evolve_skill":
            await self._execute_evolve_skill(start, operation_input)
        else:
            self._emit_fail(
                "unsupported_operation", f"Unsupported operation: {operation_name}"
            )

    async def _execute_evolve_skill(self, start: JsonObject, value: JsonValue) -> None:
        if start.get("resumeFrom") is not None:
            self._emit_fail(
                "resume_unsupported",
                "swarmx.evolve_skill does not support execution checkpoint resume.",
            )
            return
        request = _as_mapping(value, "evolve_skill request")
        validate_optimization_request(request)
        optimizer = _required_mapping(request, "optimizer")
        optimizer_id = _required_string(optimizer, "optimizerId")
        try:
            if optimizer_id == "deterministic.v1":
                await asyncio.to_thread(
                    self._run_deterministic_optimizer, start, request
                )
                return
            if optimizer_id == "dspy.gepa.v1":
                await asyncio.to_thread(self._run_gepa_optimizer, start, request)
                return
            self._emit_fail(
                "unsupported_optimizer",
                f"Unsupported skill optimizer: {optimizer_id}",
            )
        except CapabilityCancelledError:
            self._emit_canceled(self._last_checkpoint_id)
        except CapabilityError as error:
            self._emit_fail("capability_error", str(error), False)

    def _run_deterministic_optimizer(
        self, start: JsonObject, request: JsonObject
    ) -> None:
        client = _required_capability_client(self)
        client.assert_optimizer_config_digest(
            request, deterministic_config_digest(request)
        )
        baseline = _succeeded_value(
            client.call(
                "skill_evolution",
                "read_artifact",
                {"ref": _required_string(request, "baselineContentRef")},
            ),
            "read baseline",
        )
        train = _succeeded_value(
            client.call(
                "skill_evolution",
                "read_artifact",
                {"ref": _required_mapping(request, "trainDataset")["contentRef"]},
            ),
            "read train dataset",
        )
        keyword = deterministic_keyword(train.get("content"))
        baseline_text = _required_string(baseline, "content")
        candidate_text = deterministic_candidate(baseline_text, keyword)
        self._thread_emit(
            "progress", {"message": "Deterministic optimizer produced a candidate."}
        )
        artifact_path, content_digest = self._write_candidate_artifact(
            start, candidate_text
        )
        manifest = candidate_manifest_from_request(
            request, content_digest, optimizer_id="deterministic.v1"
        )
        report = {
            "optimizerId": "deterministic.v1",
            "optimizerVersion": "1",
            "actualOptimizer": "deterministic.v1",
            "seed": _required_mapping(request, "optimizer").get("seed", 0),
            "keyword": keyword,
            "modelCalls": 0,
            "tokens": 0,
            "wallTimeMs": 0,
            "errors": [],
        }
        self._thread_emit(
            "complete",
            {
                "idempotencyKey": f"complete:{self._run_id()}",
                "summary": "Deterministic skill optimization completed.",
                "result": {"candidateManifest": manifest, "optimizerReport": report},
                "artifactIds": [self._last_artifact_id],
                "checkpointId": self._last_checkpoint_id,
            },
        )

    def _run_gepa_optimizer(self, start: JsonObject, request: JsonObject) -> None:
        client = _required_capability_client(self)
        budget = _required_mapping(request, "budget")
        max_model_calls = budget.get("maxModelCalls")
        if (
            not isinstance(max_model_calls, int)
            or isinstance(max_model_calls, bool)
            or max_model_calls <= 0
        ):
            raise CapabilityError(
                "dspy.gepa.v1 requires a positive maxModelCalls budget; none was granted."
            )
        try:
            from swarmx.rsi import client as rsi_client
        except ImportError as error:
            self._emit_fail(
                "optimizer_environment_unavailable",
                f"dspy.gepa.v1 requires the locked RSI MCP environment: {error}",
            )
            return
        artifacts: dict[str, str] = {}
        for ref in (
            _required_string(request, "baselineContentRef"),
            _required_string(_required_mapping(request, "trainDataset"), "contentRef"),
            _required_string(_required_mapping(request, "devDataset"), "contentRef"),
        ):
            value = _succeeded_value(
                client.call("skill_evolution", "read_artifact", {"ref": ref}),
                "read RSI artifact",
            )
            artifacts[ref] = _required_string(value, "content")
        self._thread_emit(
            "checkpoint",
            {
                "idempotencyKey": f"checkpoint:{self._run_id()}:rsi-mcp:started",
                "checkpoint": {
                    "checkpointId": f"ckp_{self._run_id()}_rsi_mcp_started",
                    "format": "swarmx.python.evolve_skill",
                    "formatVersion": 1,
                    "environmentDigest": self.environment_digest,
                    "state": {"phase": "rsi_mcp_started"},
                },
            },
        )
        try:
            report, candidate_text = rsi_client.run_rsi_optimizer(
                request=request,
                artifacts=artifacts,
                capability_client=client,
                cancel_check=lambda: self.cancel_event.is_set(),
                progress=lambda message, fraction: self._thread_emit(
                    "progress", {"message": message, "fraction": fraction}
                ),
            )
        except rsi_client.RsiCancelledError as error:
            raise CapabilityCancelledError(str(error)) from error
        except rsi_client.RsiError as error:
            raise CapabilityError(str(error)) from error
        artifact_path, content_digest = self._write_candidate_artifact(
            start, candidate_text
        )
        manifest = candidate_manifest_from_request(
            request, content_digest, optimizer_id=report["optimizerId"]
        )
        self._thread_emit(
            "complete",
            {
                "idempotencyKey": f"complete:{self._run_id()}",
                "summary": "GEPA skill optimization completed.",
                "result": {"candidateManifest": manifest, "optimizerReport": report},
                "artifactIds": [self._last_artifact_id],
                "checkpointId": self._last_checkpoint_id,
            },
        )

    def _write_candidate_artifact(
        self, start: JsonObject, candidate_text: str
    ) -> tuple[str, str]:
        request = _required_mapping(start, "operation")
        request_id = str(
            _required_mapping(request, "input").get("requestId", "candidate")
        )
        relative_path = f"candidates/{request_id}.md"
        if not is_safe_relative_path(relative_path):
            raise ProtocolError("candidate artifact path is not a safe relative path")
        encoded = candidate_text.encode("utf-8")
        if len(encoded) > 4 * 1024 * 1024:
            self._emit_fail(
                "candidate_oversized", "Optimizer produced an oversized candidate."
            )
            raise CapabilityError("candidate oversized")
        os.makedirs("candidates", exist_ok=True)
        with open(relative_path, "wb") as handle:
            handle.write(encoded)
        content_digest = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
        artifact_id = f"art_evolve_{uuid.uuid4().hex[:12]}"
        self._last_artifact_id = artifact_id
        checkpoint_id = f"ckp_{self._run_id()}_evolve"
        self._last_checkpoint_id = checkpoint_id
        self._thread_emit(
            "checkpoint",
            {
                "idempotencyKey": f"checkpoint:{self._run_id()}:evolve",
                "checkpoint": {
                    "checkpointId": checkpoint_id,
                    "format": "swarmx.python.evolve_skill",
                    "formatVersion": 1,
                    "environmentDigest": self.environment_digest,
                    "state": {"phase": "exported", "candidateDigest": content_digest},
                },
            },
        )
        self._thread_emit(
            "artifact",
            {
                "idempotencyKey": f"artifact:{self._run_id()}:candidate",
                "artifact": {
                    "artifactId": artifact_id,
                    "kind": "skill_candidate",
                    "relativePath": relative_path,
                    "sha256": content_digest,
                    "sizeBytes": len(encoded),
                    "mediaType": "text/markdown",
                },
            },
        )
        return relative_path, content_digest

    def _thread_emit(self, message_type: str, payload: JsonObject) -> None:
        client = getattr(self, "capability_client", None)
        loop = client._loop if client is not None else asyncio.get_event_loop()

        async def emit() -> None:
            self._emit(message_type, payload)

        future = asyncio.run_coroutine_threadsafe(emit(), loop)
        future.result(timeout=30)

    def _capability_budget(self, message: JsonObject) -> dict[str, int]:
        budget = message.get("budget")
        if not isinstance(budget, dict):
            return {}
        calls = budget.get("capabilityCalls")
        if not isinstance(calls, dict):
            return {}
        return {
            str(key): int(value)
            for key, value in calls.items()
            if isinstance(value, int)
        }

    def _request_token_budget(self, message: JsonObject) -> int | None:
        operation = message.get("operation")
        if not isinstance(operation, dict):
            return None
        request = operation.get("input")
        if not isinstance(request, dict):
            return None
        budget = request.get("budget")
        if not isinstance(budget, dict):
            return None
        max_tokens = budget.get("maxTokens")
        if not isinstance(max_tokens, int) or max_tokens < 0:
            return None
        return max_tokens

    async def _execute_echo(self, value: JsonValue) -> None:
        if self.cancel_event.is_set():
            self._emit_canceled()
            return
        self._emit("heartbeat", {})
        self._emit(
            "progress", {"message": "Echo operation completed.", "fraction": 1.0}
        )
        checkpoint_id = self._checkpoint_id(1)
        self._emit(
            "checkpoint",
            {
                "idempotencyKey": f"checkpoint:{self._run_id()}:{checkpoint_id}",
                "checkpoint": {
                    "checkpointId": checkpoint_id,
                    "format": "swarmx.python.echo",
                    "formatVersion": 1,
                    "environmentDigest": self.environment_digest,
                    "state": {"completed": True, "value": value},
                },
            },
        )
        self._emit(
            "complete",
            {
                "idempotencyKey": f"complete:{self._run_id()}",
                "summary": "Python echo operation completed.",
                "result": value,
                "artifactIds": [],
                "checkpointId": checkpoint_id,
            },
        )

    async def _execute_count(self, start: JsonObject, value: JsonValue) -> None:
        operation_input = _mapping_or_empty(value)
        steps = _bounded_integer(operation_input.get("steps", 3), "steps", 1, 10_000)
        delay_ms = _bounded_integer(
            operation_input.get("delayMs", 0), "delayMs", 0, 60_000
        )
        next_step, last_checkpoint_id = self._resume_position(start, steps)
        resumed_from = next_step

        while next_step < steps:
            if await self._cancelled_during_delay(delay_ms):
                self._emit_canceled(last_checkpoint_id)
                return
            self._emit("heartbeat", {})
            next_step += 1
            self._emit(
                "progress",
                {
                    "message": f"Completed step {next_step} of {steps}.",
                    "fraction": next_step / steps,
                    "counters": {"completedSteps": next_step, "totalSteps": steps},
                },
            )
            checkpoint_id = self._checkpoint_id(next_step)
            self._emit(
                "checkpoint",
                {
                    "idempotencyKey": f"checkpoint:{self._run_id()}:{next_step}",
                    "checkpoint": {
                        "checkpointId": checkpoint_id,
                        "format": "swarmx.python.count",
                        "formatVersion": 1,
                        "environmentDigest": self.environment_digest,
                        "state": {"nextStep": next_step, "totalSteps": steps},
                    },
                },
            )
            last_checkpoint_id = checkpoint_id

        self._emit(
            "complete",
            {
                "idempotencyKey": f"complete:{self._run_id()}",
                "summary": f"Python count operation completed {steps} steps.",
                "result": {"count": steps, "resumedFrom": resumed_from},
                "artifactIds": [],
                "checkpointId": last_checkpoint_id,
            },
        )

    async def _execute_failure(self, value: JsonValue) -> None:
        operation_input = _mapping_or_empty(value)
        message = operation_input.get("message", "Requested Python worker failure.")
        retryable = operation_input.get("retryable", False)
        if not isinstance(message, str) or not isinstance(retryable, bool):
            raise ProtocolError(
                "swarmx.fail requires string message and boolean retryable"
            )
        self._emit_fail("requested_failure", message, retryable)

    async def _execute_needs_human(self, start: JsonObject, value: JsonValue) -> None:
        operation_input = _mapping_or_empty(value)
        prompt = operation_input.get("prompt", "Continue this task?")
        if not isinstance(prompt, str) or not prompt:
            raise ProtocolError("swarmx.needs_human requires a non-empty prompt")
        decisions = start.get("humanDecisions")
        if isinstance(decisions, list) and decisions:
            decision = _as_mapping(decisions[-1], "human decision")
            status = _required_string(decision, "status")
            if status in {"approved", "waived"}:
                self._emit(
                    "complete",
                    {
                        "idempotencyKey": f"complete:{self._run_id()}",
                        "summary": "Python operation resumed after a human decision.",
                        "result": {
                            "approvalId": _required_string(decision, "approvalId"),
                            "status": status,
                            "response": decision.get("response"),
                        },
                        "artifactIds": [],
                    },
                )
                return
            self._emit_fail("approval_rejected", "The human approval was rejected.")
            return
        checkpoint_id = self._checkpoint_id(0)
        self._emit(
            "checkpoint",
            {
                "idempotencyKey": f"checkpoint:{self._run_id()}:human",
                "checkpoint": {
                    "checkpointId": checkpoint_id,
                    "format": "swarmx.python.needs_human",
                    "formatVersion": 1,
                    "environmentDigest": self.environment_digest,
                    "state": {"awaitingHuman": True},
                },
            },
        )
        self._emit(
            "needs_human",
            {
                "idempotencyKey": f"needs-human:{self._run_id()}",
                "request": {
                    "requestId": f"human:{self._run_id()}",
                    "kind": "approval",
                    "prompt": prompt,
                    "options": [
                        {"optionId": "continue", "label": "Continue"},
                        {"optionId": "cancel", "label": "Cancel"},
                    ],
                    "checkpointId": checkpoint_id,
                },
            },
        )

    def _emit_fail(self, code: str, message: str, retryable: bool = False) -> None:
        self._emit(
            "fail",
            {
                "idempotencyKey": f"fail:{self._run_id()}",
                "failure": {"code": code, "message": message, "retryable": retryable},
            },
        )

    def _emit_canceled(self, checkpoint_id: str | None = None) -> None:
        payload: JsonObject = {
            "idempotencyKey": f"canceled:{self._run_id()}",
            "mode": self.cancel_mode,
            "reason": self.cancel_reason,
        }
        if checkpoint_id:
            payload["checkpointId"] = checkpoint_id
        self._emit("canceled", payload)

    def _emit(self, message_type: str, payload: JsonObject) -> None:
        active = self.active_run
        if active is None:
            raise ProtocolError("cannot emit a run event without an active run")
        message: JsonObject = {
            "protocolVersion": PROTOCOL_VERSION,
            "messageId": self._message_id(message_type),
            "direction": "worker_to_host",
            "type": message_type,
            "workItemId": _required_string(active, "workItemId"),
            "runId": _required_string(active, "runId"),
            "leaseId": _required_string(active, "leaseId"),
            "fencingToken": _required_integer(active, "fencingToken"),
            "sequence": self.sequence,
            "emittedAt": _utc_now(),
            **payload,
        }
        self.sequence += 1
        self._write(message)

    def _resume_position(self, start: JsonObject, steps: int) -> tuple[int, str | None]:
        resume_from = start.get("resumeFrom")
        if resume_from is None:
            return 0, None
        checkpoint = _as_mapping(resume_from, "resumeFrom")
        if _required_string(checkpoint, "environmentDigest") != self.environment_digest:
            raise ProtocolError(
                "checkpoint environment digest does not match this worker"
            )
        state = _required_mapping(checkpoint, "state")
        next_step = _bounded_integer(state.get("nextStep", 0), "nextStep", 0, steps)
        total_steps = _bounded_integer(
            state.get("totalSteps", steps), "totalSteps", 1, 10_000
        )
        if total_steps != steps:
            raise ProtocolError("checkpoint totalSteps does not match operation input")
        return next_step, _required_string(checkpoint, "checkpointId")

    async def _cancelled_during_delay(self, delay_ms: int) -> bool:
        if self.cancel_event.is_set():
            return True
        if delay_ms == 0:
            await asyncio.sleep(0)
            return self.cancel_event.is_set()
        try:
            await asyncio.wait_for(self.cancel_event.wait(), timeout=delay_ms / 1_000)
            return True
        except TimeoutError:
            return False

    async def _heartbeat_loop(self) -> None:
        try:
            while self.active_task and not self.active_task.done():
                await asyncio.sleep(self.heartbeat_interval_ms / 1_000)
                if self.active_task and not self.active_task.done() and self.active_run:
                    self._emit("heartbeat", {})
        except asyncio.CancelledError:
            pass

    def _matches_active_run(self, message: Mapping[str, JsonValue]) -> bool:
        active = self.active_run
        if active is None:
            return False
        return all(
            message.get(key) == active.get(key)
            for key in ("workItemId", "runId", "leaseId", "fencingToken")
        )

    def _task_finished(self, task: asyncio.Task[None]) -> None:
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as error:
            try:
                self._emit_fail("worker_exception", str(error))
            except Exception as emit_error:
                print(
                    f"SwarmX worker failed to report an exception: {emit_error}",
                    file=sys.stderr,
                )
        finally:
            self.active_task = None
            self.heartbeat_task = None
            self.active_run = None
            self.capability_client = None
            self._last_artifact_id = None
            self._last_checkpoint_id = None

    async def _stop_active_task(self) -> None:
        if self.heartbeat_task and not self.heartbeat_task.done():
            self.heartbeat_task.cancel()
        if not self.active_task or self.active_task.done():
            return
        self.active_task.cancel()
        try:
            await self.active_task
        except asyncio.CancelledError:
            pass

    def _parse_control_line(self, line: bytes) -> JsonObject:
        if len(line) > MAX_LINE_BYTES:
            raise ProtocolError(f"JSONL line exceeds {MAX_LINE_BYTES} bytes")
        if not line.endswith(b"\n"):
            raise ProtocolError(
                "control messages must be newline-terminated JSONL records"
            )
        try:
            decoded = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ProtocolError(f"invalid JSON: {error}") from error
        message = _as_mapping(decoded, "control message")
        if _required_integer(message, "protocolVersion") != PROTOCOL_VERSION:
            raise ProtocolError("unsupported protocol version")
        if _required_string(message, "direction") != "host_to_worker":
            raise ProtocolError("control message has the wrong direction")
        _required_string(message, "messageId")
        return message

    def _run_id(self) -> str:
        if self.active_run is None:
            raise ProtocolError("no active run")
        return _required_string(self.active_run, "runId")

    def _checkpoint_id(self, step: int) -> str:
        return f"ckp_{self._run_id()}_{step}"

    def _message_id(self, message_type: str) -> str:
        return f"{message_type}:{uuid.uuid4().hex}"

    @staticmethod
    def _write(message: JsonObject) -> None:
        encoded = json.dumps(message, ensure_ascii=False, separators=(",", ":"))
        if len(encoded.encode("utf-8")) > MAX_LINE_BYTES:
            raise ProtocolError(f"outbound JSONL line exceeds {MAX_LINE_BYTES} bytes")
        _PROTOCOL_STDOUT.write(f"{encoded}\n")
        _PROTOCOL_STDOUT.flush()


class CapabilityClient:
    """Grant-checked capability calls with correlation, timeout, cancel, and budget.

    Safe to call from worker threads: emissions are scheduled on the event loop
    so protocol sequence and write ordering stay single-writer; results arrive
    via `deliver_result` from the loop thread and wake the waiting caller.
    """

    def __init__(
        self,
        runtime: WorkerRuntime,
        grant_ids: list[JsonValue],
        budget: dict[str, int],
        run_id: str,
        loop: asyncio.AbstractEventLoop,
        token_budget: int | None = None,
    ) -> None:
        self._runtime = runtime
        self._grant_ids = {
            str(grant_id) for grant_id in grant_ids if isinstance(grant_id, str)
        }
        self._budget = budget
        self._token_budget = token_budget
        self._run_id = run_id
        self._used: dict[str, int] = {}
        self._used_tokens = 0
        self._pending: dict[str, dict[str, object]] = {}
        self._lock = threading.Lock()
        self._loop = loop
        self._default_timeout_ms = 60_000

    def assert_optimizer_config_digest(
        self, request: JsonObject, actual_digest: str
    ) -> None:
        optimizer = _required_mapping(request, "optimizer")
        expected = _required_string(optimizer, "configDigest")
        if not ENVIRONMENT_DIGEST_PATTERN.fullmatch(actual_digest):
            raise ProtocolError(
                "optimizer config digest must be sha256:<64 lowercase hex>"
            )
        if actual_digest != expected:
            raise CapabilityError(
                f"Optimizer config digest {actual_digest} does not match the granted {expected}."
            )

    def call(
        self,
        capability_id: str,
        operation: str,
        arguments: JsonObject,
        timeout_ms: int | None = None,
    ) -> JsonObject:
        self._check_budget(capability_id, operation)
        self._check_token_budget(capability_id, operation)
        arguments = self._clamp_max_tokens(operation, arguments)
        call_id = f"cap:{uuid.uuid4().hex}"
        idempotency_key = f"cap:{self._run_id}:{operation}:{call_id}"
        timeout_ms = timeout_ms or self._default_timeout_ms
        event = threading.Event()
        with self._lock:
            self._pending[call_id] = {"event": event, "outcome": None}

        async def emit_call() -> None:
            self._runtime._emit(
                "capability_call",
                {
                    "callId": call_id,
                    "grantId": self._grant(),
                    "capabilityId": capability_id,
                    "operation": operation,
                    "idempotencyKey": idempotency_key,
                    "arguments": arguments,
                },
            )

        emission = asyncio.run_coroutine_threadsafe(emit_call(), self._loop)
        emission.result(timeout=30)

        deadline = time.monotonic() + timeout_ms / 1_000
        while True:
            if self._runtime.cancel_event.is_set():
                with self._lock:
                    self._pending.pop(call_id, None)
                raise CapabilityCancelledError("Capability call cancelled by the host.")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                with self._lock:
                    self._pending.pop(call_id, None)
                raise CapabilityTimeoutError(f"Capability call {operation} timed out.")
            if event.wait(timeout=min(remaining, 0.25)):
                with self._lock:
                    pending = self._pending.pop(call_id, None)
                outcome = pending["outcome"] if pending else None
                if isinstance(outcome, dict):
                    self._charge_tokens(outcome)
                    return outcome
                raise CapabilityError("Missing capability outcome.")

    def deliver_result(self, message: JsonObject) -> None:
        call_id = _required_string(message, "callId")
        outcome = message.get("outcome")
        if not isinstance(outcome, dict):
            outcome = {
                "status": "unknown",
                "error": {
                    "code": "malformed_result",
                    "message": "Capability result is malformed.",
                    "retryable": False,
                },
            }
        with self._lock:
            pending = self._pending.get(call_id)
            if pending is None:
                return
            pending["outcome"] = outcome
            event = pending["event"]
        if isinstance(event, threading.Event):
            event.set()

    def _check_budget(self, capability_id: str, operation: str) -> None:
        key = f"{capability_id}:{operation}"
        with self._lock:
            used = self._used.get(key, 0) + 1
            limit = self._budget.get(key)
            if limit is not None and used > limit:
                raise CapabilityBudgetError(
                    f"Capability call budget exhausted for {key} (limit {limit})."
                )
            self._used[key] = used

    def _check_token_budget(self, capability_id: str, operation: str) -> None:
        if self._token_budget is None or operation != "model.generate":
            return
        with self._lock:
            if self._token_budget == 0:
                raise CapabilityBudgetError(
                    "The granted model token budget is zero; model calls are denied."
                )
            if self._used_tokens >= self._token_budget:
                raise CapabilityBudgetError(
                    f"Model token budget exhausted ({self._used_tokens} >= {self._token_budget})."
                )

    def _clamp_max_tokens(self, operation: str, arguments: JsonObject) -> JsonObject:
        if operation != "model.generate" or self._token_budget is None:
            return arguments
        requested = arguments.get("maxTokens")
        if not isinstance(requested, int) or requested <= 0:
            return arguments
        with self._lock:
            remaining = max(0, self._token_budget - self._used_tokens)
        if remaining <= 0:
            raise CapabilityBudgetError(
                f"Model token budget exhausted ({self._used_tokens} >= {self._token_budget})."
            )
        if requested > remaining:
            clamped = dict(arguments)
            clamped["maxTokens"] = remaining
            return clamped
        return arguments

    def _charge_tokens(self, outcome: JsonObject) -> None:
        if self._token_budget is None:
            return
        value = outcome.get("value")
        usage = value.get("usage") if isinstance(value, dict) else None
        token_value = usage.get("totalTokens", 0) if isinstance(usage, dict) else 0
        tokens = (
            token_value
            if isinstance(token_value, int) and not isinstance(token_value, bool)
            else 0
        )
        if tokens <= 0:
            return
        with self._lock:
            self._used_tokens += tokens
            if self._used_tokens > self._token_budget:
                raise CapabilityBudgetError(
                    f"Model token budget exhausted ({self._used_tokens} > {self._token_budget})."
                )

    def _grant(self) -> str:
        if not self._grant_ids:
            raise ProtocolError("no capability grant is bound to this run")
        return sorted(self._grant_ids)[0]


def validate_optimization_request(request: JsonObject) -> None:
    if _required_integer(request, "schemaVersion") != 1:
        raise ProtocolError("evolve_skill request schemaVersion must be 1")
    for key in request:
        if not REQUEST_KEY_PATTERN.fullmatch(str(key)):
            raise ProtocolError(f"evolve_skill request key {key!r} is unsafe")
    known_keys = {
        "schemaVersion",
        "skillId",
        "variantId",
        "parentRevisionId",
        "parentRevisionDigest",
        "baselineContentRef",
        "baselineContentDigest",
        "targetAgentId",
        "targetModelFingerprint",
        "trainDataset",
        "devDataset",
        "optimizer",
        "budget",
        "requestedBy",
        "requestId",
        "proposer",
    }
    unknown = set(request) - known_keys
    if unknown:
        raise ProtocolError(
            f"evolve_skill request contains unauthorized fields: {sorted(unknown)}"
        )
    proposer = request.get("proposer", "none")
    if proposer not in {"none", "gateway", "deterministic"}:
        raise ProtocolError(f"unsupported optimizer proposer mode: {proposer}")
    _required_string(request, "baselineContentRef")
    _required_string(request, "parentRevisionDigest")
    dataset = _required_mapping(request, "trainDataset")
    if _required_string(dataset, "role") != "train":
        raise ProtocolError("trainDataset role must be train")
    dev = _required_mapping(request, "devDataset")
    if _required_string(dev, "role") != "dev":
        raise ProtocolError("devDataset role must be dev")
    optimizer = _required_mapping(request, "optimizer")
    _required_string(optimizer, "optimizerId")
    _required_string(optimizer, "configDigest")
    _required_integer(optimizer, "seed")
    budget = _required_mapping(request, "budget")
    max_wall = budget.get("maxWallTimeMs")
    if isinstance(max_wall, int) and not 1 <= max_wall <= 7 * 24 * 3600 * 1000:
        raise ProtocolError("maxWallTimeMs is out of bounds")
    max_artifact = budget.get("maxArtifactBytes")
    if isinstance(max_artifact, int) and not 1 <= max_artifact <= 16 * 1024 * 1024:
        raise ProtocolError("maxArtifactBytes is out of bounds")


def deterministic_keyword(train_content: object) -> str:
    if isinstance(train_content, str):
        try:
            records = json.loads(train_content)
            if isinstance(records, list):
                for record in records:
                    if isinstance(record, dict) and isinstance(
                        record.get("keyword"), str
                    ):
                        return record["keyword"]
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    digest = hashlib.sha256(str(train_content).encode("utf-8")).hexdigest()
    return f"kw-{digest[:6]}"


def deterministic_candidate(baseline_text: str, keyword: str) -> str:
    fragment = f"\n\n### Optimized guidance\n\nAlways include the word `{keyword}` in your final answer. This rule is mandatory.\n"
    if keyword in baseline_text:
        return baseline_text
    return baseline_text + fragment


def deterministic_config_digest(request: JsonObject) -> str:
    optimizer = _required_mapping(request, "optimizer")
    seed = _required_integer(optimizer, "seed")
    budget = _required_mapping(request, "budget")
    canonical = json.dumps(
        {
            "schemaVersion": 1,
            "optimizerId": "deterministic.v1",
            "seed": seed,
            "proposer": request.get("proposer", "none"),
            "budget": {
                "maxModelCalls": budget.get("maxModelCalls"),
                "maxTokens": budget.get("maxTokens"),
                "maxWallTimeMs": budget.get("maxWallTimeMs"),
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}"


def candidate_manifest_from_request(
    request: JsonObject, content_digest: str, optimizer_id: str
) -> JsonObject:
    optimizer = _required_mapping(request, "optimizer")
    seed = _required_integer(optimizer, "seed")
    revision_id = f"r_{content_digest[len('sha256:') :]}"
    return {
        "skillId": _required_string(request, "skillId"),
        "variantId": _required_string(request, "variantId"),
        "revisionId": revision_id,
        "parentRevisionId": _required_string(request, "parentRevisionId"),
        "parentRevisionDigest": _required_string(request, "parentRevisionDigest"),
        "contentDigest": content_digest,
        "mediaType": "text/markdown",
        "targetAgentId": _required_string(request, "targetAgentId"),
        "targetModelFingerprint": _required_string(request, "targetModelFingerprint"),
        "optimizer": {
            "optimizerId": _required_string(optimizer, "optimizerId"),
            "optimizerVersion": _required_string(optimizer, "optimizerVersion"),
            "environmentDigest": _required_string(optimizer, "environmentDigest"),
            "configDigest": _required_string(optimizer, "configDigest"),
            "seed": seed,
        },
        "trainDatasetDigest": _required_mapping(request, "trainDataset").get(
            "contentDigest"
        ),
        "devDatasetDigest": _required_mapping(request, "devDataset").get(
            "contentDigest"
        ),
    }


def is_safe_relative_path(value: str) -> bool:
    if (
        value.startswith("/")
        or value.startswith("\\")
        or re.match(r"^[A-Za-z]:[\\/]", value)
    ):
        return False
    segments = value.replace("\\", "/").split("/")
    return all(segment not in ("", ".", "..") for segment in segments)


def _required_capability_client(runtime: WorkerRuntime) -> CapabilityClient:
    client = runtime.capability_client
    if client is None:
        raise ProtocolError("no capability client is bound to this run")
    return client


def _succeeded_value(outcome: JsonObject, label: str) -> JsonObject:
    if outcome.get("status") != "succeeded":
        error = outcome.get("error")
        message = (
            str(error.get("message"))
            if isinstance(error, dict)
            else "capability failed"
        )
        raise CapabilityError(f"{label} failed: {message}")
    value = outcome.get("value")
    return _as_mapping(value, f"{label} value")


def _as_mapping(value: object, label: str) -> JsonObject:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ProtocolError(f"{label} must be a JSON object")
    return value


def _required_mapping(value: Mapping[str, JsonValue], key: str) -> JsonObject:
    return _as_mapping(value.get(key), key)


def _mapping_or_empty(value: JsonValue) -> JsonObject:
    if value is None:
        return {}
    return _as_mapping(value, "operation input")


def _required_string(value: Mapping[str, JsonValue], key: str) -> str:
    field = value.get(key)
    if not isinstance(field, str) or not field:
        raise ProtocolError(f"{key} must be a non-empty string")
    return field


def _required_integer(value: Mapping[str, JsonValue], key: str) -> int:
    field = value.get(key)
    if not isinstance(field, int) or isinstance(field, bool):
        raise ProtocolError(f"{key} must be an integer")
    return field


def _bounded_integer(value: JsonValue, label: str, minimum: int, maximum: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ProtocolError(
            f"{label} must be an integer from {minimum} through {maximum}"
        )
    return value


def _utc_now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SwarmX Python task worker.")
    parser.add_argument("--environment-digest", required=True)
    args = parser.parse_args(argv)
    if not ENVIRONMENT_DIGEST_PATTERN.fullmatch(args.environment_digest):
        parser.error(
            "--environment-digest must be sha256:<64 lowercase hex characters>"
        )
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    return asyncio.run(WorkerRuntime(args.environment_digest).run())


if __name__ == "__main__":
    raise SystemExit(main())
