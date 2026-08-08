"""Unit tests for the Inspect skill paired-eval adapter."""

import hashlib
import json
import unittest
from pathlib import Path

from skill_eval import (
    assert_eval_safe_config,
    build_evidence_file,
    evidence_from_metadata,
    score_run,
    seeded_bit,
    string_or_none,
)


class SeededBitTest(unittest.TestCase):
    def test_is_deterministic(self) -> None:
        self.assertEqual(seeded_bit(7, "h1"), seeded_bit(7, "h1"))
        self.assertIn(seeded_bit(7, "h1"), (True, False))

    def test_differs_across_case_ids(self) -> None:
        bits = {seeded_bit(7, f"h{i}") for i in range(8)}
        self.assertEqual(len(bits), 2)


class ScoreRunTest(unittest.TestCase):
    def test_scores_target_match_not_any_output(self) -> None:
        # A non-empty output that misses the target must not pass.
        self.assertEqual(
            score_run({"output": "nope", "latencyMs": 5}, "parrot", None, None)[
                "passed"
            ],
            False,
        )
        self.assertEqual(
            score_run({"output": "say parrot", "latencyMs": 5}, "parrot", None, None)[
                "passed"
            ],
            True,
        )

    def test_safety_and_failure(self) -> None:
        run = score_run(
            {"output": "UNSAFE text", "latencyMs": 5}, "parrot", None, "UNSAFE"
        )
        self.assertFalse(run["safetyPassed"])
        self.assertEqual(
            score_run({"output": "", "error": "boom", "latencyMs": 5}, "x", None, None)[
                "failed"
            ],
            True,
        )

    def test_context_tokens_from_eval_metrics(self) -> None:
        run = score_run(
            {"output": "parrot", "latencyMs": 5, "metrics": {"contextTokens": 110}},
            "parrot",
            None,
            None,
        )
        self.assertEqual(run["contextTokens"], 110)
        self.assertEqual(
            score_run({"output": "parrot", "latencyMs": 5}, "parrot", None, None)[
                "contextTokens"
            ],
            0,
        )


class EvidenceConversionTest(unittest.TestCase):
    def test_builds_core_evidence_shape(self) -> None:
        records = [
            {
                "metadata": {
                    "swarmx_paired": {
                        "caseId": "h1",
                        "candidateRanFirst": True,
                        "sample": {
                            "caseId": "h1",
                            "candidateRanFirst": True,
                            "baseline": {
                                "passed": False,
                                "safetyPassed": True,
                                "contextTokens": 0,
                                "latencyMs": 2,
                                "failed": False,
                            },
                            "candidate": {
                                "passed": True,
                                "safetyPassed": True,
                                "contextTokens": 110,
                                "latencyMs": 3,
                                "failed": False,
                            },
                        },
                    }
                }
            }
        ]
        samples = evidence_from_metadata(records)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["caseId"], "h1")
        self.assertTrue(samples[0]["candidate"]["passed"])
        self.assertFalse(samples[0]["baseline"]["passed"])
        self.assertEqual(samples[0]["candidate"]["contextTokens"], 110)

    def test_build_evidence_file_writes_digest_and_counts(self) -> None:
        output_path = "/tmp/swarmx-skill-eval-evidence-test.json"
        digest = hashlib.sha256(b"holdout").hexdigest()
        build_evidence_file(
            samples=[],
            baseline_revision="r_a",
            candidate_revision="r_b",
            holdout_digest=f"sha256:{digest}",
            holdout_case_count=4,
            seed=3,
            output_path=output_path,
        )
        evidence = json.loads(Path(output_path).read_text(encoding="utf-8"))
        self.assertEqual(evidence["evaluatorId"], "inspect.skill_paired_eval")
        self.assertEqual(evidence["holdoutCaseCount"], 4)
        self.assertEqual(evidence["candidateRevisionId"], "r_b")
        self.assertIn("samples", evidence)


class EvalSafeConfigTest(unittest.TestCase):
    def test_rejects_side_effect_surfaces(self) -> None:
        safe = json.dumps(
            {
                "name": "safe",
                "root": "agent",
                "nodes": {"agent": {"kind": "agent", "agent": {"name": "agent"}}},
                "edges": [],
            }
        ).encode("utf-8")
        assert_eval_safe_config(safe)
        cases: list[dict[str, object]] = [
            {"queen": {"name": "q"}},
            {"mcpServers": {"s": {"url": "http://x"}}},
            {"hooks": [{"event": "pre_run"}]},
            {"nodes": {"t": {"kind": "tool", "tool": {"name": "t"}}}},
            {
                "nodes": {
                    "a": {
                        "kind": "agent",
                        "agent": {
                            "name": "a",
                            "backend": {"type": "custom", "program": "x"},
                        },
                    }
                }
            },
        ]
        for patch in cases:
            config: dict[str, object] = {
                "name": "safe",
                "root": "agent",
                "nodes": {"agent": {"kind": "agent", "agent": {"name": "agent"}}},
                "edges": [],
            }
            config.update(patch)
            with self.assertRaises(ValueError):
                assert_eval_safe_config(json.dumps(config).encode("utf-8"))


class StringOrNoneTest(unittest.TestCase):
    def test_strings_and_none(self) -> None:
        self.assertEqual(string_or_none("x"), "x")
        self.assertIsNone(string_or_none(42))
        self.assertIsNone(string_or_none(None))


if __name__ == "__main__":
    unittest.main()
