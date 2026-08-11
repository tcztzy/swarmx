"""Acceptance tests for the bounded read-only Reference contract."""

from __future__ import annotations

import unittest

from swarmx.ref.service import (
    ReferenceService,
    ZoteroReferenceBackend,
    html_to_text,
)


class FakeBackend:
    def __init__(self, kind: str = "zim") -> None:
        self.kind = kind

    def status(self) -> dict:
        return {"kind": self.kind, "name": f"{self.kind} fixture"}

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
        self.service = ReferenceService({"zim": FakeBackend()})

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
            self.service.handle(
                {"operation": "search", "source": "unknown", "query": "x"}
            )
        with self.assertRaises(ValueError):
            self.service.handle({"operation": "search", "source": "web", "query": "x"})
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

    def test_status_lists_configured_sources_without_querying_them(self) -> None:
        self.assertEqual(
            [
                source["id"]
                for source in self.service.handle({"operation": "status"})["sources"]
            ],
            ["zim"],
        )


class ZoteroReferenceBackendTest(unittest.TestCase):
    def test_reads_only_top_level_item_metadata(self) -> None:
        requested: list[str] = []

        def fetch_json(url: str) -> tuple[object, dict[str, str]]:
            requested.append(url)
            if "/items/top?" in url:
                return (
                    [
                        {
                            "key": "ABCD2345",
                            "data": {
                                "itemType": "journalArticle",
                                "title": "A paper",
                                "creators": [
                                    {"firstName": "Ada", "lastName": "Lovelace"}
                                ],
                                "date": "1843",
                                "abstractNote": "An <i>objective</i> abstract.",
                                "publicationTitle": "Notes",
                                "DOI": "10.1/example",
                                "url": "https://example.com/paper",
                            },
                        }
                    ],
                    {"Total-Results": "4"},
                )
            return (
                {
                    "key": "ABCD2345",
                    "data": {
                        "itemType": "journalArticle",
                        "title": "A paper",
                        "creators": [{"name": "Ada Lovelace"}],
                        "date": "1843",
                        "abstractNote": "An <i>objective</i> abstract.",
                    },
                },
                {},
            )

        service = ReferenceService(
            {"zotero": ZoteroReferenceBackend(fetch_json=fetch_json)}
        )
        searched = service.handle(
            {
                "operation": "search",
                "source": "zotero",
                "query": "paper",
                "limit": 1,
            }
        )
        self.assertEqual(searched["estimatedMatches"], 4)
        self.assertEqual(searched["matches"][0]["path"], "ABCD2345")
        item = service.handle(
            {"operation": "get", "source": "zotero", "path": "ABCD2345"}
        )
        self.assertIn("Ada Lovelace", item["text"])
        self.assertIn("An objective abstract.", item["text"])
        self.assertTrue(
            all(
                "/items/top" in url or url.endswith("/items/ABCD2345")
                for url in requested
            )
        )

    def test_rejects_non_item_keys(self) -> None:
        backend = ZoteroReferenceBackend(fetch_json=lambda _url: ({}, {}))
        with self.assertRaises(ValueError):
            backend.get("../fulltext", 100)


if __name__ == "__main__":
    unittest.main()
