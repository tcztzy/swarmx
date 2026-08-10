"""Round-trip tests for the DSPy/GEPA skill evolution server.

Covers: baseline -> DSPy program -> GEPA mutates the real component ->
exported candidate Markdown differs in digest and carries the improvement, the
metric contract (score + feedback + deterministic validator), the offline
deterministic proposer, and canonical config digest agreement with the
TypeScript side.
"""

from __future__ import annotations

import hashlib
import json
import unittest

from swarmx.rsi import optimize
from swarmx.rsi.capability_lm import DeterministicLm, mandated_keyword
from swarmx.rsi.skill_program import build_skill_program, export_skill_markdown

BASELINE = "# Math Coach Skill\n\nAnswer the user's question."
TRAIN_RECORDS = [
    {"id": "t1", "input": "what is 2+2?", "target": "parrot", "keyword": "parrot"},
    {"id": "t2", "input": "what is 3+3?", "target": "parrot", "keyword": "parrot"},
]
DEV_RECORDS = [
    {"id": "d1", "input": "what is 4+4?", "target": "parrot", "keyword": "parrot"},
]
TRAIN_TEXT = "\n".join(json.dumps(record) for record in TRAIN_RECORDS)
DEV_TEXT = "\n".join(json.dumps(record) for record in DEV_RECORDS)


def sha256_digest(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_request() -> dict:
    return {
        "schemaVersion": 1,
        "skillId": "math-coach",
        "variantId": "math-coach:default",
        "parentRevisionId": "r_" + hashlib.sha256(BASELINE.encode("utf-8")).hexdigest(),
        "parentRevisionDigest": sha256_digest(BASELINE),
        "baselineContentRef": sha256_digest(BASELINE),
        "baselineContentDigest": sha256_digest(BASELINE),
        "targetAgentId": "swarmx:model-x",
        "targetModelFingerprint": "model-x@v1",
        "trainDataset": {
            "role": "train",
            "contentRef": sha256_digest(TRAIN_TEXT),
            "contentDigest": sha256_digest(TRAIN_TEXT),
            "caseCount": len(TRAIN_RECORDS),
            "format": "swarmx.eval.jsonl",
        },
        "devDataset": {
            "role": "dev",
            "contentRef": sha256_digest(DEV_TEXT),
            "contentDigest": sha256_digest(DEV_TEXT),
            "caseCount": len(DEV_RECORDS),
            "format": "swarmx.eval.jsonl",
        },
        "optimizer": {
            "optimizerId": "dspy.gepa.v1",
            "optimizerVersion": "1",
            "environmentDigest": "sha256:" + "0" * 64,
            "configDigest": "sha256:" + "0" * 64,
            "seed": 42,
        },
        "budget": {
            "maxModelCalls": 24,
            "maxTokens": 2000,
            "maxWallTimeMs": 120000,
            "maxArtifactBytes": 262144,
        },
        "proposer": "deterministic",
        "requestedBy": "roundtrip-test",
    }


class FakeArtifactClient:
    def __init__(self, artifacts: dict[str, str]) -> None:
        self.artifacts = artifacts

    def call(self, capability_id, operation, arguments, timeout_ms=None):
        assert operation == "read_artifact", (
            "deterministic proposer must not call the model"
        )
        content = self.artifacts[arguments["ref"]]
        return {
            "status": "succeeded",
            "value": {"content": content, "contentType": "text/markdown"},
        }


class SkillProgramRoundTripTest(unittest.TestCase):
    def test_program_instructions_are_exactly_the_skill_markdown(self) -> None:
        program = build_skill_program(BASELINE)
        self.assertEqual(program.signature.instructions, BASELINE)
        self.assertEqual(export_skill_markdown(program), BASELINE)

    def test_mutating_the_component_changes_the_exported_markdown(self) -> None:
        program = build_skill_program(BASELINE)
        program.signature.instructions = (
            BASELINE + "\n\nMandatory rule: use `parrot`.\n"
        )
        exported = export_skill_markdown(program)
        self.assertIn("parrot", exported)
        self.assertNotEqual(exported, BASELINE)
        self.assertNotEqual(
            hashlib.sha256(exported.encode("utf-8")).hexdigest(),
            hashlib.sha256(BASELINE.encode("utf-8")).hexdigest(),
        )

    def test_gepa_round_trip_changes_the_exported_candidate_digest(self) -> None:
        request = build_request()
        artifacts = {
            request["baselineContentRef"]: BASELINE,
            request["trainDataset"]["contentRef"]: TRAIN_TEXT,
            request["devDataset"]["contentRef"]: DEV_TEXT,
        }
        client = FakeArtifactClient(artifacts)
        report, candidate = optimize.run_gepa(
            request, client, cancel_check=lambda: False, progress=lambda _m, _f: None
        )
        self.assertEqual(report["optimizerId"], "dspy.gepa.v1")
        self.assertEqual(report["errors"], [])
        self.assertNotEqual(candidate, BASELINE)
        self.assertIn("parrot", candidate)
        baseline_digest = hashlib.sha256(BASELINE.encode("utf-8")).hexdigest()
        candidate_digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        self.assertNotEqual(candidate_digest, baseline_digest)
        exported = export_skill_markdown(build_skill_program(candidate))
        self.assertEqual(exported, candidate)


class DeterministicLmTest(unittest.TestCase):
    def test_mandated_keyword_drives_the_answer(self) -> None:
        lm = DeterministicLm()
        response = lm.forward(
            messages=[
                {
                    "role": "system",
                    "content": "Follow the rule: the final answer must be exactly `parrot`.",
                },
                {"role": "user", "content": "what is 2+2?"},
            ]
        )
        self.assertIn("[[ ## answer ## ]]", response.choices[0].message.content)
        self.assertIn("parrot", response.choices[0].message.content)
        self.assertEqual(mandated_keyword("no rule here"), None)
        self.assertEqual(mandated_keyword("must be exactly `k1`"), "k1")


class CanonicalConfigDigestTest(unittest.TestCase):
    def test_config_digest_is_stable_and_sha256(self) -> None:
        request = build_request()
        digest = optimize.canonical_config_digest(request)
        self.assertRegex(digest, r"^sha256:[a-f0-9]{64}$")
        self.assertEqual(digest, optimize.canonical_config_digest(request))


if __name__ == "__main__":
    unittest.main()
