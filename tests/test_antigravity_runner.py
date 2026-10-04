import io
import json
import unittest
from unittest.mock import Mock, patch

from demo.antigravity_runner import AntigravityActionResolver, AntigravityChatSession, CachedActionExecutor


class AntigravityActionResolverTests(unittest.TestCase):
    @patch("demo.antigravity_runner.subprocess.run")
    def test_resolve_unwraps_current_cli_structured_output(self, run):
        run.return_value = Mock(
            returncode=0,
            stdout=json.dumps({
                "status": "SUCCESS",
                "structured_output": {"status": "ready", "verified": True},
            }),
            stderr="",
        )
        resolver = AntigravityActionResolver(".")
        self.assertEqual(resolver.resolve("open youtube"), {"status": "ready", "verified": True})

    def test_command_contains_schema_and_optional_permission_flag(self):
        resolver = AntigravityActionResolver(".", allow_unattended_tools=True)
        with patch.object(resolver, "find_executable", return_value="agy.exe"):
            command = resolver.build_command("Mở YouTube")
        self.assertIn("--dangerously-skip-permissions", command)
        self.assertIn("--json-schema", command)
        self.assertIn("--output-format", command)
        self.assertIn("USER_DESCRIPTION", command[-3])

    @patch("demo.antigravity_runner.subprocess.Popen")
    def test_cached_executor_uses_fallback(self, popen_mock):
        popen_mock.side_effect = [OSError("missing"), object()]
        profile = {
            "status": "ready",
            "verified": True,
            "primary": {"type": "open_app", "executable": "missing.exe", "arguments": [], "working_directory": ""},
            "fallbacks": [{"type": "open_app", "executable": "working.exe", "arguments": [], "working_directory": ""}],
        }
        result = CachedActionExecutor().execute(profile)
        self.assertTrue(result.success)
        self.assertEqual(1, result.selected_index)

    def test_cached_executor_rejects_unverified_profile(self):
        result = CachedActionExecutor().execute({"status": "ready", "verified": False})
        self.assertFalse(result.success)

    @patch("demo.antigravity_runner.threading.Thread")
    @patch("demo.antigravity_runner.subprocess.Popen")
    @patch("demo.antigravity_runner.AntigravityActionResolver.find_executable", return_value="agy.exe")
    def test_chat_session_uses_stream_json_and_sends_ndjson(self, _find_mock, popen_mock, thread_mock):
        process = popen_mock.return_value
        process.poll.return_value = None
        process.stdin = io.StringIO()
        process.stdout = io.StringIO()
        process.stderr = io.StringIO()
        session = AntigravityChatSession(".")
        session.send("Xin chào")
        command = popen_mock.call_args.args[0]
        self.assertIn("stream-json", command)
        payload = json.loads(process.stdin.getvalue())
        self.assertEqual("user", payload["event"])
        self.assertEqual("Xin chào", payload["message"])


if __name__ == "__main__":
    unittest.main()
