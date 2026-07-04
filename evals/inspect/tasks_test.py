import unittest

from tasks import sample_config


class InspectTasksTest(unittest.TestCase):
    def test_sample_config_prefers_string_metadata_config(self) -> None:
        self.assertEqual(
            sample_config({"config": "echo.json"}, "default.json"), "echo.json"
        )

    def test_sample_config_ignores_non_string_metadata_config(self) -> None:
        self.assertEqual(sample_config({"config": 42}, "default.json"), "default.json")


if __name__ == "__main__":
    unittest.main()
