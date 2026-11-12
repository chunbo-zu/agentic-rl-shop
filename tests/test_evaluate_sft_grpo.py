"""Tests for the SFT/GRPO evaluation orchestrator."""

import subprocess
import unittest
from pathlib import Path
from urllib.error import HTTPError, URLError
from unittest.mock import patch

from scripts.evaluate_sft_grpo import _http_ready


ROOT = Path(__file__).resolve().parents[1]


class EvaluationServiceReadinessTest(unittest.TestCase):
    def test_http_404_means_server_is_listening(self):
        error = HTTPError(
            "http://127.0.0.1:5700/",
            404,
            "Not Found",
            hdrs=None,
            fp=None,
        )
        with patch("scripts.evaluate_sft_grpo.urlopen", side_effect=error):
            self.assertTrue(_http_ready("http://127.0.0.1:5700"))

    def test_connection_failure_is_not_ready(self):
        with patch(
            "scripts.evaluate_sft_grpo.urlopen",
            side_effect=URLError("connection refused"),
        ):
            self.assertFalse(_http_ready("http://127.0.0.1:5700"))

    def test_model_launcher_exposes_virtualenv_tools(self):
        launcher = (ROOT / "scripts/serve_model.sh").read_text(encoding="utf-8")
        self.assertIn('export PATH="$VENV_BIN:$PATH"', launcher)
        self.assertIn("command -v ninja", launcher)
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts/serve_model.sh")],
            check=True,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
