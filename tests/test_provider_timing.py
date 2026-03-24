"""Test that the provider captures TTFT/decode timing in TokenUsage."""
from unittest.mock import MagicMock, patch

from claw_eval.models.content import TextBlock
from claw_eval.models.message import Message
from claw_eval.runner.providers.openai_compat import OpenAICompatProvider


def _make_stream_chunks(content_parts: list[str]):
    """Create mock SSE stream chunks."""
    chunks = []
    for text in content_parts:
        chunk = MagicMock()
        chunk.usage = None
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = text
        chunk.choices[0].delta.tool_calls = None
        chunk.choices[0].delta.reasoning_content = None
        chunk.choices[0].delta.reasoning = None
        chunks.append(chunk)

    # Final chunk with usage
    final = MagicMock()
    final.choices = []
    final.usage = MagicMock()
    final.usage.prompt_tokens = 100
    final.usage.completion_tokens = len(content_parts)
    chunks.append(final)

    return chunks


def test_streaming_captures_ttft():
    provider = OpenAICompatProvider(model_id="test-model", api_key="fake")
    chunks = _make_stream_chunks(["Hello", " world", "!"])

    with patch.object(provider.client.chat.completions, "create", return_value=iter(chunks)):
        kwargs = {"model": "test-model", "messages": [{"role": "user", "content": "Hi"}]}
        resp = provider._call_with_stream(kwargs)

    msg, usage = provider._parse_response(resp)
    assert usage.ttft_ms is not None
    assert usage.ttft_ms >= 0
    assert usage.content_tokens == 3
    assert usage.decode_ms is not None
    assert usage.decode_ms >= 0


def test_non_streaming_has_no_ttft():
    provider = OpenAICompatProvider(model_id="test-model", api_key="fake")

    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message = MagicMock()
    response.choices[0].message.content = "Hello world"
    response.choices[0].message.tool_calls = None
    response.choices[0].message.reasoning_content = None
    response.choices[0].message.reasoning = None
    response.usage = MagicMock()
    response.usage.prompt_tokens = 50
    response.usage.completion_tokens = 2
    response._timing = None  # non-streaming: no timing data

    msg, usage = provider._parse_response(response)
    # Non-streaming: no TTFT data
    assert usage.ttft_ms is None
    assert usage.content_tokens is None
