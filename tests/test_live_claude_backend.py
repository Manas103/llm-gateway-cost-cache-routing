import random
import shutil

import pytest

from llmgw.backends import ClaudeCLIBackend

pytestmark = pytest.mark.skipif(shutil.which("claude") is None, reason="claude CLI not on PATH")


def test_claude_cli_backend_returns_a_real_response():
    backend = ClaudeCLIBackend()
    result = backend.complete("Reply with exactly one word: pong", 10, 0, "easy", random.Random(1))
    assert result.output_tokens >= 1
    assert result.latency_ms > 0
