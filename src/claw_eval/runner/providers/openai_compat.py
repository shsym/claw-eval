"""OpenAI-compatible provider (GPT-4o, vLLM, MiMo, etc.)."""

from __future__ import annotations

import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from uuid import uuid4
from typing import Any

from openai import OpenAI

from ...models.content import AudioBlock, ImageBlock, TextBlock, ToolUseBlock, VideoBlock
from ...models.message import Message
from ...models.tool import ToolSpec
from ...models.trace import TokenUsage


def _tool_spec_to_openai(spec: ToolSpec) -> dict[str, Any]:
    """Convert our ToolSpec to OpenAI function calling format."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.input_schema,
        },
    }


def _audio_format_from_mime(mime_type: str) -> str:
    """Map mime type to OpenAI input_audio format."""
    mime = mime_type.lower()
    if mime in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return "wav"
    if mime in {"audio/mp3", "audio/mpeg"}:
        return "mp3"
    return "wav"


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    flags=re.IGNORECASE | re.DOTALL,
)
_FUNCTION_RE = re.compile(
    r"<function\s*=\s*([a-zA-Z0-9_:-]+)\s*>",
    flags=re.IGNORECASE,
)
_PARAM_RE = re.compile(
    r"<parameter\s*=\s*([a-zA-Z0-9_:-]+)\s*>(.*?)</parameter>",
    flags=re.IGNORECASE | re.DOTALL,
)


def _coerce_param_value(raw: str) -> Any:
    value = raw.strip()
    if not value:
        return ""

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None

    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except Exception:
            return value
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except Exception:
            return value

    if (value.startswith("{") and value.endswith("}")) or (
        value.startswith("[") and value.endswith("]")
    ):
        try:
            return json.loads(value)
        except Exception:
            return value

    return value


def _extract_text_tool_calls(text: str) -> tuple[str, list[ToolUseBlock]]:
    """Parse pseudo tool-call markup from text as fallback.

    This supports common non-native formats like:
    <tool_call>
      <function=todo_list_tasks>
      <parameter=status>all</parameter>
    </tool_call>
    """
    tool_uses: list[ToolUseBlock] = []
    if "<tool_call" not in text.lower():
        return text, tool_uses

    matches = list(_TOOL_CALL_BLOCK_RE.finditer(text))
    if not matches:
        return text, tool_uses

    for m in matches:
        block = m.group(1)
        fn = _FUNCTION_RE.search(block)
        if not fn:
            continue

        tool_name = fn.group(1).strip()
        parsed_input: dict[str, Any] = {}
        for p in _PARAM_RE.finditer(block):
            key = p.group(1).strip()
            raw_val = p.group(2)
            parsed_input[key] = _coerce_param_value(raw_val)

        tool_uses.append(
            ToolUseBlock(
                id=f"fallback_{uuid4().hex[:12]}",
                name=tool_name,
                input=parsed_input,
            )
        )

    if not tool_uses:
        return text, tool_uses

    cleaned_text = _TOOL_CALL_BLOCK_RE.sub("", text).strip()
    return cleaned_text, tool_uses


class _PiePromptCacheClient:
    """Thin client for the inferlet's phase-1 prefix-cache endpoints."""

    def __init__(self, base_url: str | None, config: dict[str, Any] | None) -> None:
        self.enabled = bool(config and config.get("enabled", True) and base_url)
        self.schema_id = (config or {}).get("schema_id", "claw-eval.system-message")
        self.schema_version = (config or {}).get("schema_version", "1")
        self.prompt_mode = (config or {}).get("prompt_mode", "default")
        self.on_epoch_mismatch = (config or {}).get("on_epoch_mismatch", "render_full_request")
        self.on_missing_prefix = (config or {}).get("on_missing_prefix", "render_full_request")
        self.timeout_s = float((config or {}).get("timeout_s", 5.0))
        self._base_url = (base_url or "").rstrip("/")
        if self._base_url.endswith("/v1"):
            self._base_url = self._base_url[:-3]
        self._cache_epoch: str | None = None
        self._handles_by_prefix: dict[str, str] = {}

    def build_extra_body(self, messages: list[Message]) -> dict[str, Any] | None:
        if not self.enabled:
            return None

        prefix_text = self._extract_prefix_text(messages)
        if not prefix_text:
            return None

        caps = self._request_json("GET", "/v1/pie/prompt-cache/capabilities")
        if not isinstance(caps, dict):
            return None
        features = caps.get("features") or {}
        cache = caps.get("cache") or {}
        if not features.get("registered_prefix_blocks"):
            return None

        epoch = cache.get("epoch")
        if not isinstance(epoch, str) or not epoch:
            return None
        if epoch != self._cache_epoch:
            self._cache_epoch = epoch
            self._handles_by_prefix.clear()

        prefix_handle = self._handles_by_prefix.get(prefix_text)
        if prefix_handle is None:
            ensured = self._request_json(
                "POST",
                "/v1/pie/prompt-cache/prefixes:ensure",
                {
                    "schema": {
                        "schema_id": self.schema_id,
                        "schema_version": self.schema_version,
                    },
                    "prefix": {
                        "prompt_mode": self.prompt_mode,
                        "slots": [
                            {
                                "placement": "system.base",
                                "content": {"format": "text", "text": prefix_text},
                            }
                        ],
                    },
                    "fallback": {"on_prefix_miss": self.on_missing_prefix},
                    "retention": {"scope": "instance", "priority": "warm"},
                },
            )
            if not isinstance(ensured, dict):
                return None
            prefix_handle = ensured.get("prefix_handle")
            cache = ensured.get("cache") or {}
            if not isinstance(prefix_handle, str) or not prefix_handle:
                return None
            if isinstance(cache.get("epoch"), str) and cache["epoch"] != self._cache_epoch:
                self._cache_epoch = cache["epoch"]
                self._handles_by_prefix.clear()
            self._handles_by_prefix[prefix_text] = prefix_handle

        return {
            "pie_prompt": {
                "version": "1",
                "mode": "registered_prefix",
                "cache_epoch": self._cache_epoch,
                "prefix_handle": prefix_handle,
                "fallback": {
                    "on_epoch_mismatch": self.on_epoch_mismatch,
                    "on_missing_prefix": self.on_missing_prefix,
                },
            }
        }

    def _extract_prefix_text(self, messages: list[Message]) -> str | None:
        if not messages:
            return None
        first = messages[0]
        if first.role != "system":
            return None
        text = first.text.strip()
        return text or None

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        url = f"{self._base_url}{path}"
        data = None
        headers = {"User-Agent": "claw-eval/1.0"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return None


def _blocks_to_openai_content(msg: Message) -> str | list[dict[str, Any]]:
    """Convert text/image/audio/video blocks into OpenAI content parts.

    Returns plain string for text-only content to preserve compatibility.
    """
    text_parts = [b.text for b in msg.content if b.type == "text"]
    has_non_text = any(b.type in {"image", "audio", "video"} for b in msg.content)
    if not has_non_text:
        return "\n".join(text_parts) if text_parts else ""

    parts: list[dict[str, Any]] = []
    for block in msg.content:
        if block.type == "text":
            parts.append({"type": "text", "text": block.text})
        elif block.type == "image":
            block = block if isinstance(block, ImageBlock) else ImageBlock.model_validate(block)
            parts.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:{block.mime_type};base64,{block.data}",
                },
            })
        elif block.type == "audio":
            block = block if isinstance(block, AudioBlock) else AudioBlock.model_validate(block)
            parts.append({
                "type": "input_audio",
                "input_audio": {
                    "data": block.data,
                    "format": _audio_format_from_mime(block.mime_type),
                },
            })
        elif block.type == "video":
            block = block if isinstance(block, VideoBlock) else VideoBlock.model_validate(block)
            # Most OpenAI-compatible chat endpoints do not universally support
            # native video parts yet; convert to explicit text marker.
            parts.append({
                "type": "text",
                "text": (
                    f"[video attached: {block.source_path or 'inline'} "
                    f"({block.mime_type}, base64_bytes={len(block.data) * 3 // 4})]"
                ),
            })
    return parts


def _message_to_openai(msg: Message) -> dict[str, Any] | list[dict[str, Any]]:
    """Convert our Message to OpenAI chat format.

    Returns a single dict for simple messages, or a list of dicts
    when tool_result blocks need to be sent as separate tool messages.
    """
    # Tool result messages need special handling
    tool_results = [b for b in msg.content if b.type == "tool_result"]
    if tool_results:
        results = []
        for tr in tool_results:
            content_text = "\n".join(t.text for t in tr.content) if tr.content else ""
            results.append({
                "role": "tool",
                "tool_call_id": tr.tool_use_id,
                "content": content_text,
            })
        return results

    # Assistant messages with tool_use blocks
    tool_uses = [b for b in msg.content if b.type == "tool_use"]
    if tool_uses:
        d = {
            "role": "assistant",
            "content": _blocks_to_openai_content(msg),
            "tool_calls": [
                {
                    "id": tu.id,
                    "type": "function",
                    "function": {
                        "name": tu.name,
                        "arguments": json.dumps(tu.input),
                    },
                }
                for tu in tool_uses
            ],
        }
        if msg.reasoning_content:
            # Use "reasoning" for OpenRouter compatibility (also accepted as
            # "reasoning_content" by native DeepSeek/QwQ endpoints).
            d["reasoning"] = msg.reasoning_content
        return d

    # Simple text message
    d = {
        "role": msg.role,
        "content": _blocks_to_openai_content(msg),
    }
    if msg.reasoning_content:
        d["reasoning"] = msg.reasoning_content
    return d


class OpenAICompatProvider:
    """Calls any OpenAI-compatible chat completions endpoint."""

    def __init__(
        self,
        model_id: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        extra_body: dict | None = None,
    ) -> None:
        self.model_id = model_id
        extra = dict(extra_body or {})
        self.force_stream = extra.pop("force_stream", False)
        prompt_cache_cfg = extra.pop("pie_prompt_cache", None)
        self.extra_body = extra
        self.prompt_cache = _PiePromptCacheClient(base_url, prompt_cache_cfg)
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY") or "unused"
        self.client = OpenAI(
            api_key=resolved_key,
            base_url=base_url,
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[ToolSpec] | None = None,
    ) -> tuple[Message, TokenUsage]:
        """Send messages to the model and return parsed response."""
        has_multimodal_input = any(
            b.type in {"image", "audio", "video"}
            for m in messages
            for b in m.content
        )

        # Build OpenAI messages list
        oai_messages: list[dict[str, Any]] = []
        for msg in messages:
            converted = _message_to_openai(msg)
            if isinstance(converted, list):
                oai_messages.extend(converted)
            else:
                oai_messages.append(converted)

        kwargs: dict[str, Any] = {
            "model": self.model_id,
            "messages": oai_messages,
            "temperature": 0.0,
        }
        extra_body = dict(self.extra_body)
        prompt_cache_extra = self.prompt_cache.build_extra_body(messages)
        if prompt_cache_extra:
            extra_body.update(prompt_cache_extra)
        if extra_body:
            kwargs["extra_body"] = extra_body
        if tools:
            kwargs["tools"] = [_tool_spec_to_openai(t) for t in tools]

        max_retries = 5
        last_exc: Exception | None = None
        use_stream = bool(self.force_stream)  # force_stream=True enables TTFT capture
        for attempt in range(max_retries + 1):
            try:
                if use_stream or attempt > 1:
                    response = self._call_with_stream(kwargs)
                else:
                    response = self._call_without_stream(kwargs)
                # Parse inside try so that empty-choices errors are retried
                return self._parse_response(response)
            except Exception as exc:
                last_exc = exc
                # Check if retryable (rate-limit or server error)
                status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
                exc_str = str(exc).lower()
                exc_type = type(exc).__name__.lower()
                retryable = (
                    status in (429, 500, 502, 503, 529)
                    or "timeout" in exc_str
                    or "connection" in exc_str
                    or "empty choices" in exc_str
                    or "remoteprotocol" in exc_type
                    or "remoteprotocol" in exc_str
                    or "peer closed" in exc_str
                    or "incomplete" in exc_str
                    or "server disconnected" in exc_str
                )
                if not retryable or attempt == max_retries:
                    if has_multimodal_input:
                        raise RuntimeError(
                            "Model endpoint rejected multimodal input. "
                            "Check provider support for image/audio/video message parts, "
                            "or set media.strict_mode=false to allow skips."
                        ) from exc
                    raise
                # Non-streaming timeout → switch to streaming to keep connection alive
                if not use_stream and ("timeout" in exc_str or "read timed out" in exc_str) and attempt >= 1:
                    use_stream = True
                    print(f"[retry] Switching to streaming mode after repeated timeouts")
                # Exponential backoff with jitter
                delay = random.uniform(2, 4)
                print(f"[retry] API error ({status or type(exc).__name__}), "
                      f"attempt {attempt + 1}/{max_retries}, waiting {delay:.1f}s ...")
                time.sleep(delay)

        # All retries exhausted (should not normally reach here)
        raise last_exc or RuntimeError("All retries exhausted")

    # ------------------------------------------------------------------
    # Non-streaming call (fallback for endpoints where streaming 502s)
    # ------------------------------------------------------------------

    def _call_without_stream(self, kwargs: dict[str, Any]) -> Any:
        """Call the API without streaming. Used as fallback when streaming fails."""
        return self.client.chat.completions.create(**kwargs)

    # ------------------------------------------------------------------
    # Streaming call + chunk assembly
    # ------------------------------------------------------------------

    def _call_with_stream(self, kwargs: dict[str, Any]) -> Any:
        """Call the API with streaming to avoid reverse-proxy read timeouts.

        Assembles a synthetic response object that looks like a non-stream
        response so the rest of the code path stays the same.
        """
        stream_kwargs: dict[str, Any] = {"stream": True}
        # stream_options is OpenAI-specific; Anthropic/Claude endpoints reject it.
        model_lower = kwargs.get("model", "").lower()
        is_anthropic = any(s in model_lower for s in ("claude", "anthropic"))
        if not is_anthropic:
            stream_kwargs["stream_options"] = {"include_usage": True}
        stream = self.client.chat.completions.create(**kwargs, **stream_kwargs)

        # Timing instrumentation
        stream_start = time.monotonic()
        first_content_ts: float | None = None
        content_token_count = 0

        reasoning_parts: list[str] = []
        content_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        usage_info = None
        has_any_choice = False

        for chunk in stream:
            if chunk.usage:
                usage_info = chunk.usage
            if not chunk.choices:
                continue
            has_any_choice = True
            delta = chunk.choices[0].delta

            # reasoning_content (thinking models: DeepSeek-R1, QwQ, etc.)
            # OpenRouter returns "reasoning" instead of "reasoning_content"
            rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
            if rc:
                reasoning_parts.append(rc)

            # content
            if delta.content:
                if first_content_ts is None:
                    first_content_ts = time.monotonic()
                content_parts.append(delta.content)
                content_token_count += 1

            # tool_calls (streamed incrementally)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_by_index:
                        tool_calls_by_index[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_by_index[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_by_index[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_by_index[idx]["arguments"] += tc_delta.function.arguments

        if not has_any_choice:
            raise RuntimeError("Model returned empty choices (choices=None or [])")

        # Compute timing metrics
        stream_end = time.monotonic()
        ttft_ms = ((first_content_ts - stream_start) * 1000) if first_content_ts is not None else None
        decode_ms = ((stream_end - first_content_ts) * 1000) if first_content_ts is not None else None
        ms_per_token = (
            round(decode_ms / content_token_count, 2)
            if decode_ms is not None and content_token_count > 0
            else None
        )

        # Build a lightweight object that _parse_response can consume
        # (duck-typed to match openai ChatCompletion structure).
        class _Msg:
            pass

        msg = _Msg()
        msg.content = "".join(content_parts) if content_parts else None
        msg.reasoning_content = "".join(reasoning_parts) if reasoning_parts else None

        if tool_calls_by_index:
            assembled = []
            for idx in sorted(tool_calls_by_index):
                tc = tool_calls_by_index[idx]

                class _Fn:
                    pass

                fn = _Fn()
                fn.name = tc["name"]
                fn.arguments = tc["arguments"]

                class _TC:
                    pass

                t = _TC()
                t.id = tc["id"]
                t.function = fn
                assembled.append(t)
            msg.tool_calls = assembled
        else:
            msg.tool_calls = None

        class _Choice:
            pass

        choice = _Choice()
        choice.message = msg

        class _Resp:
            pass

        resp = _Resp()
        resp.choices = [choice]
        resp.usage = usage_info

        # Attach timing data for _parse_response to pick up
        resp._timing = {
            "ttft_ms": round(ttft_ms, 2) if ttft_ms is not None else None,
            "decode_ms": round(decode_ms, 2) if decode_ms is not None else None,
            "content_tokens": content_token_count if content_token_count > 0 else None,
            "ms_per_token": ms_per_token,
        }
        return resp

    # ------------------------------------------------------------------
    # Response parsing (shared by stream / non-stream paths)
    # ------------------------------------------------------------------

    def _parse_response(self, response: Any) -> tuple[Message, TokenUsage]:
        """Parse a (possibly synthetic) ChatCompletion into Message + TokenUsage."""
        if not response.choices:
            raise RuntimeError("Model returned empty choices (choices=None or [])")
        choice = response.choices[0]

        # Parse response into our content blocks
        content_blocks = []
        if choice.message.content:
            if isinstance(choice.message.content, str):
                content_blocks.append(TextBlock(text=choice.message.content))
            elif isinstance(choice.message.content, list):
                text_chunks = []
                for part in choice.message.content:
                    if isinstance(part, dict):
                        part_type = part.get("type")
                        if part_type == "text":
                            text_val = part.get("text")
                            if isinstance(text_val, str):
                                text_chunks.append(text_val)
                        continue

                    part_type = getattr(part, "type", None)
                    if part_type == "text":
                        text_val = getattr(part, "text", None)
                        if isinstance(text_val, str):
                            text_chunks.append(text_val)
                if text_chunks:
                    content_blocks.append(TextBlock(text="\n".join(text_chunks)))

        parsed_tool_uses: list[ToolUseBlock] = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                parsed_tool_uses.append(ToolUseBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                ))
            content_blocks.extend(parsed_tool_uses)
        else:
            # Fallback for providers that return pseudo tool markup in text
            # instead of native tool_calls payloads.
            text_blocks = [b for b in content_blocks if b.type == "text"]
            if text_blocks:
                merged_text = "\n".join(tb.text for tb in text_blocks)
                cleaned_text, fallback_tools = _extract_text_tool_calls(merged_text)
                if fallback_tools:
                    content_blocks = []
                    if cleaned_text:
                        content_blocks.append(TextBlock(text=cleaned_text))
                    content_blocks.extend(fallback_tools)

        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            )

        # Pick up streaming timing data if present
        timing = getattr(response, "_timing", None)
        if timing:
            usage.ttft_ms = timing.get("ttft_ms")
            usage.decode_ms = timing.get("decode_ms")
            usage.content_tokens = timing.get("content_tokens")
            usage.ms_per_token = timing.get("ms_per_token")

        # Capture reasoning_content from thinking models (DeepSeek-R1, QwQ, etc.)
        # OpenRouter returns "reasoning" instead of "reasoning_content"
        reasoning = getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)

        return Message(role="assistant", content=content_blocks, reasoning_content=reasoning), usage
