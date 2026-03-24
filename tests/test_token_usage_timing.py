"""Test that TokenUsage carries inference timing fields."""
from claw_eval.models.trace import TokenUsage


def test_token_usage_has_timing_fields():
    usage = TokenUsage(
        input_tokens=100,
        output_tokens=50,
        ttft_ms=18.4,
        decode_ms=1024.2,
        content_tokens=25,
        ms_per_token=40.97,
    )
    assert usage.ttft_ms == 18.4
    assert usage.decode_ms == 1024.2
    assert usage.content_tokens == 25
    assert usage.ms_per_token == 40.97


def test_token_usage_timing_defaults_to_none():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    assert usage.ttft_ms is None
    assert usage.decode_ms is None
    assert usage.content_tokens is None
    assert usage.ms_per_token is None


def test_token_usage_serialization_omits_none():
    usage = TokenUsage(input_tokens=10, output_tokens=5)
    d = usage.model_dump(exclude_none=True)
    assert "ttft_ms" not in d
    assert "decode_ms" not in d
