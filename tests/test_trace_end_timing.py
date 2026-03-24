"""Test that TraceEnd includes aggregated timing fields."""
from claw_eval.models.trace import TraceEnd


def test_trace_end_has_timing_aggregates():
    te = TraceEnd(
        trace_id="test",
        avg_ttft_ms=18.4,
        avg_decode_ms=1024.2,
        avg_ms_per_token=40.97,
        total_content_tokens=75,
    )
    assert te.avg_ttft_ms == 18.4
    assert te.avg_decode_ms == 1024.2
    assert te.avg_ms_per_token == 40.97
    assert te.total_content_tokens == 75


def test_trace_end_timing_defaults_to_none():
    te = TraceEnd(trace_id="test")
    assert te.avg_ttft_ms is None
    assert te.avg_decode_ms is None
    assert te.avg_ms_per_token is None
    assert te.total_content_tokens is None
