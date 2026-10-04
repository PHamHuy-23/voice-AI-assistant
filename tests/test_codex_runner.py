import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from demo.codex_runner import CodexRunner


class CodexRunnerTests(unittest.TestCase):
    workspace = str(Path(__file__).parents[1])
    log_dir = str(Path(__file__).parent / "codex-runner-test-logs")

    def test_rejects_empty_prompt(self):
        runner = CodexRunner(self.workspace, self.log_dir)
        self.assertFalse(runner.start("empty", "   "))

    @patch("demo.codex_runner.Path.write_text")
    @patch("demo.codex_runner.Path.mkdir")
    @patch("demo.codex_runner.CodexRunner.find_executable", return_value="codex.cmd")
    @patch("demo.codex_runner.subprocess.run")
    def test_runs_workspace_scoped_codex_command(self, run_mock, _find_mock, _mkdir_mock, _write_mock):
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "done"
        run_mock.return_value.stderr = ""

        finished = threading.Event()
        results = []
        runner = CodexRunner(self.workspace, self.log_dir)
        self.assertTrue(runner.start("fix_tests", "Fix the tests", on_finished=lambda r: (results.append(r), finished.set())))
        self.assertTrue(finished.wait(3))

        args = run_mock.call_args.args[0]
        self.assertIn("--approve-for-me", args)
        self.assertEqual("Fix the tests", args[-1])
        self.assertTrue(results[0].success)
        _write_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
