"""Acceptance tests for the single standard SwarmX Python package."""

from __future__ import annotations

import importlib
import importlib.metadata
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class PackageLayoutTest(unittest.TestCase):
    def test_uses_one_regular_distribution_with_direct_product_dependencies(
        self,
    ) -> None:
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

        self.assertTrue((ROOT / "src/swarmx/__init__.py").is_file())
        self.assertFalse((ROOT / "src/swarmx-rsi").exists())
        self.assertFalse((ROOT / "src/swarmx-ref").exists())
        self.assertEqual(project["project"]["name"], "swarmx")
        self.assertEqual(
            project["project"]["dependencies"],
            ["dspy==3.3.0", "libzim==3.12.0", "mcp==2.0.0"],
        )
        self.assertEqual(
            project["project"]["scripts"],
            {
                "swarmx-ref": "swarmx.ref.server:main",
                "swarmx-rsi": "swarmx.rsi.server:main",
            },
        )
        self.assertNotIn("rsi", project.get("dependency-groups", {}))
        self.assertNotIn("ref", project.get("dependency-groups", {}))
        self.assertNotIn("workspace", project.get("tool", {}).get("uv", {}))
        self.assertNotIn("sources", project.get("tool", {}).get("uv", {}))

    def test_installs_worker_and_private_servers_from_only_swarmx(self) -> None:
        for module in ("swarmx.worker", "swarmx.rsi.server", "swarmx.ref.server"):
            self.assertIsNotNone(importlib.import_module(module))

        self.assertEqual(
            importlib.metadata.version("swarmx"),
            "4.0.0",
        )
        self.assertEqual(importlib.metadata.version("mcp"), "2.0.0")
        distributions = {
            distribution.metadata["Name"]
            for distribution in importlib.metadata.distributions()
        }
        self.assertNotIn("swarmx-rsi", distributions)
        self.assertNotIn("swarmx-ref", distributions)
        self.assertEqual(
            list(importlib.metadata.entry_points().select(group="swarmx.modules")),
            [],
        )


if __name__ == "__main__":
    unittest.main()
