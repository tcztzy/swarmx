"""Acceptance tests for the bounded read-only Reference contract."""

from __future__ import annotations

import unittest

from swarmx.ref.service import ReferenceService, html_to_text


class FakeBackend:
    def status(self) -> dict:
        return {"source": {"fileName": "wikipedia.zim", "fileSize": 123}}

    def search(self, query: str, limit: int) -> dict:
        return {
            "query": query,
            "mode": "full_text",
            "estimatedMatches": 1,
            "matches": [{"path": "A/Test", "title": "Test"}][:limit],
        }

    def get(self, path: str, max_chars: int) -> dict:
        return {
            "path": path,
            "title": "Test",
            "mimeType": "text/plain",
            "text": "objective reference"[:max_chars],
            "truncated": max_chars < len("objective reference"),
        }


class ReferenceServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = ReferenceService(FakeBackend())

    def test_supports_only_bounded_read_operations(self) -> None:
        self.assertEqual(
            self.service.handle({"operation": "search", "query": "SwarmX", "limit": 1})[
                "matches"
            ][0]["title"],
            "Test",
        )
        self.assertEqual(
            self.service.handle({"operation": "get", "path": "A/Test", "maxChars": 9})[
                "text"
            ],
            "objective",
        )
        with self.assertRaises(ValueError):
            self.service.handle({"operation": "create", "title": "Opinion"})

    def test_rejects_unknown_fields_and_oversized_inputs(self) -> None:
        with self.assertRaises(ValueError):
            self.service.handle({"operation": "status", "path": "/secret"})
        with self.assertRaises(ValueError):
            self.service.handle({"operation": "search", "query": "x" * 257})
        with self.assertRaises(ValueError):
            self.service.handle({"operation": "search", "query": "x", "limit": 21})
        with self.assertRaises(ValueError):
            self.service.handle({"operation": "get", "path": "A", "maxChars": 32_001})

    def test_strips_active_html_before_returning_reference_text(self) -> None:
        self.assertEqual(
            html_to_text(
                "<h1>Fact</h1><script>steal()</script><p>Body &amp; source</p>"
            ).strip(),
            "Fact\nBody & source",
        )


if __name__ == "__main__":
    unittest.main()
