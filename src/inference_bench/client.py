"""One bounded streaming HTTP client for both serving engines."""

from __future__ import annotations

import asyncio
import codecs
import re
import time
from urllib.parse import urlsplit

from .provenance import canonical_json, digest_json, parse_json


class StreamError(ValueError):
    pass


class SSEDecoder:
    """Incremental UTF-8 and SSE framing, independent of HTTP read boundaries."""

    def __init__(self, max_event_chars: int = 1_048_576):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.buffer = ""
        self.data: list[str] = []
        self.data_size = 0
        self.limit = max_event_chars

    def feed(self, chunk: bytes, *, final: bool = False) -> list[str]:
        self.buffer += self.decoder.decode(chunk, final=final)
        events = []
        while match := re.search(r"\r\n|\r|\n", self.buffer):
            if match.group() == "\r" and match.end() == len(self.buffer) and not final:
                break
            line = self.buffer[:match.start()]
            self.buffer = self.buffer[match.end():]
            if not line:
                if self.data:
                    events.append("\n".join(self.data))
                    self.data.clear()
                    self.data_size = 0
            elif not line.startswith(":"):
                field, _, value = line.partition(":")
                if field == "data":
                    if value.startswith(" "):
                        value = value[1:]
                    self.data.append(value)
                    self.data_size += len(value)
            if self.data_size > self.limit:
                raise StreamError("SSE event exceeds the buffer limit")
        if len(self.buffer) + self.data_size > self.limit:
            raise StreamError("SSE event exceeds the buffer limit")
        if final and (self.buffer or self.data):
            raise StreamError("Stream ended inside an SSE event")
        return events


def request_body(
    prompt: dict, output_tokens: int, seed: int, *,
    ignore_eos: bool = True, stop_token_ids: list[int] | None = None,
) -> dict:
    ids = prompt.get("prompt_ids")
    if not isinstance(ids, list) or not ids or any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("A nonempty list of integer prompt IDs is required")
    if prompt.get("prompt_sha256") != digest_json(ids):
        raise ValueError("Prepared prompt checksum mismatch")
    if type(output_tokens) is not int or output_tokens < 1:
        raise ValueError("Output-token target must be positive")
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("Sampling seed must be a nonnegative signed 32-bit integer")
    if type(ignore_eos) is not bool or (ignore_eos and stop_token_ids is not None):
        raise ValueError("Fixed-length requests ignore EOS and cannot include stop tokens")
    if stop_token_ids is not None and (
        not isinstance(stop_token_ids, list) or not stop_token_ids
        or any(type(i) is not int or i < 0 for i in stop_token_ids)
    ):
        raise ValueError("Stop token IDs must be a nonempty list of nonnegative integers")
    body = {
        "model": "gemma4-bench", "prompt": ids, "max_tokens": output_tokens,
        "temperature": 0.0, "top_p": 1.0, "presence_penalty": 0.0,
        "frequency_penalty": 0.0, "repetition_penalty": 1.0,
        "seed": seed, "n": 1, "echo": False, "stop": [], "ignore_eos": ignore_eos,
        "skip_special_tokens": False, "stream": True,
        "stream_options": {"include_usage": True},
    }
    if stop_token_ids is not None:
        # Probes use the pinned model's stop IDs instead of either server's defaults.
        body["stop_token_ids"] = stop_token_ids
    return body


def http_client(concurrency: int):
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Install the client extra: pip install -e '.[client]'") from exc
    return httpx.AsyncClient(
        limits=httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency),
        trust_env=False, follow_redirects=False, http2=False,
    )


class ResponseTrace:
    """Content timestamps and usage are distinct from transport chunks."""

    def __init__(self, record: dict):
        self.record = record
        self.events: list[dict] = []
        self.text: list[str] = []
        self.finished = False
        self.usage_after_finish = False

    def accept(self, data: str, received_ns: int) -> None:
        r = self.record
        if r["done_ns"] is not None:
            raise StreamError("An event arrived after DONE")
        if data == "[DONE]":
            if not self.finished:
                raise StreamError("DONE arrived before a finish reason")
            r["done_ns"] = received_ns
            self.events.append({
                "request_id": r["request_id"], "received_ns": received_ns, "data": "[DONE]",
            })
            return
        try:
            event = parse_json(data)
        except ValueError as exc:
            raise StreamError("Invalid JSON in SSE event") from exc
        if not isinstance(event, dict) or event.get("error") is not None:
            raise StreamError("Server returned an invalid event or error")
        choices = event.get("choices", [])
        if not isinstance(choices, list) or len(choices) > 1:
            raise StreamError("Expected at most one completion choice")
        if choices:
            choice = choices[0]
            if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                raise StreamError("Unexpected completion choice")
            content = choice.get("text", "")
            if not isinstance(content, str):
                raise StreamError("Completion text is not a string")
            if content:
                if self.finished:
                    raise StreamError("Content arrived after the finish event")
                if r["first_content_ns"] is None:
                    r["first_content_ns"] = received_ns
                r["last_content_ns"] = received_ns
                r["content_event_count"] += 1
                self.text.append(content)
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str) or not reason:
                    raise StreamError("Invalid finish reason")
                if self.finished:
                    raise StreamError("Duplicate finish event")
                self.finished = True
                r["finish_reason"] = reason
        if event.get("usage") is not None:
            usage = event["usage"]
            if not isinstance(usage, dict):
                raise StreamError("Invalid token usage")
            for source, target in (
                ("prompt_tokens", "actual_input_tokens"),
                ("completion_tokens", "actual_output_tokens"),
            ):
                value = usage.get(source)
                if type(value) is not int or value < 0:
                    raise StreamError("Missing or invalid authoritative token usage")
                r[target] = value
            total = usage.get("total_tokens")
            if total is not None and total != r["actual_input_tokens"] + r["actual_output_tokens"]:
                raise StreamError("Inconsistent total-token usage")
            self.usage_after_finish = self.finished
        self.events.append({
            "request_id": r["request_id"], "received_ns": received_ns, "data": event,
        })

    def validate(self, exact_output: bool = True) -> None:
        r = self.record
        if r["done_ns"] is None:
            raise StreamError("Stream ended without DONE")
        if not self.usage_after_finish:
            raise StreamError("Missing final authoritative usage")
        if r["actual_input_tokens"] != r["expected_input_tokens"]:
            raise StreamError("Server input usage differs from prepared length")
        if exact_output and r["actual_output_tokens"] != r["requested_output_tokens"]:
            raise StreamError("Server output usage differs from the fixed target")
        if not exact_output and not 1 <= r["actual_output_tokens"] <= r["requested_output_tokens"]:
            raise StreamError("Server output usage lies outside the requested limit")
        if r["first_content_ns"] is None:
            raise StreamError("No streamed content was observable")


def _new_record(prompt: dict, ordinal: int, output_tokens: int, phase: str, context: dict) -> dict:
    return {
        **context, "schema_version": 1, "request_id": prompt["request_id"],
        "sequence_ordinal": ordinal, "prompt_hash": prompt["prompt_sha256"], "phase": phase,
        "expected_input_tokens": len(prompt["prompt_ids"]), "requested_output_tokens": output_tokens,
        "dispatch_ns": None, "headers_ns": None, "first_content_ns": None,
        "last_content_ns": None, "done_ns": None, "error_ns": None,
        "http_status": None, "finish_reason": None, "actual_input_tokens": None,
        "actual_output_tokens": None, "content_event_count": 0,
        "status": "pending", "error_category": None, "error_message": None,
        "response_text": "",
    }


async def run_cell(
    base_url: str, prompts: list[dict], *, concurrency: int, output_tokens: int,
    seed: int, timeout_s: float = 300, phase: str = "measurement",
    context: dict | None = None, client=None, ignore_eos: bool = True,
    stop_token_ids: list[int] | None = None,
) -> dict:
    """Stop new dispatch after a failure; drain existing work without retries."""
    url = urlsplit(base_url)
    if (
        url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1")
        or url.username or url.password or url.query or url.fragment or url.path not in ("", "/")
    ):
        raise ValueError("Measurements require a plain loopback HTTP origin")
    if not prompts or type(concurrency) is not int or not 1 <= concurrency <= len(prompts):
        raise ValueError("Concurrency must be positive and no larger than the request count")
    if timeout_s <= 0 or phase not in ("measurement", "warmup", "pilot", "conformance"):
        raise ValueError("Invalid timeout or phase")
    if phase != "conformance" and (ignore_eos is not True or stop_token_ids is not None):
        raise ValueError("Only conformance probes may honor EOS or stop tokens")
    ids = [p["request_id"] for p in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("Request IDs must be unique within a cell")
    payloads = [
        canonical_json(request_body(
            p, output_tokens, seed + i, ignore_eos=ignore_eos, stop_token_ids=stop_token_ids,
        )).encode("utf-8")
        for i, p in enumerate(prompts)
    ]
    if client is None:
        async with http_client(concurrency) as owned_client:
            return await _run(
                base_url, prompts, payloads, concurrency, output_tokens, timeout_s,
                phase, context or {}, owned_client, seed, ignore_eos,
            )
    return await _run(
        base_url, prompts, payloads, concurrency, output_tokens, timeout_s,
        phase, context or {}, client, seed, ignore_eos,
    )


async def _run(
    base_url, prompts, payloads, concurrency, output_tokens, timeout_s, phase, context, client, seed,
    exact_output,
):
    import httpx

    queue = iter(enumerate(prompts))
    stop = asyncio.Event()
    records = []
    events = []
    loop_lag_ms = []
    sampling_done = asyncio.Event()

    async def sample_lag():
        loop = asyncio.get_running_loop()
        interval = 0.01
        while not sampling_done.is_set():
            due = loop.time() + interval
            await asyncio.sleep(interval)
            loop_lag_ms.append(max(0.0, (loop.time() - due) * 1000))

    async def worker():
        while not stop.is_set():
            try:
                ordinal, prompt = next(queue)
            except StopIteration:
                return
            record = _new_record(prompt, ordinal, output_tokens, phase, context)
            record["sampling_seed"] = seed + ordinal
            trace = ResponseTrace(record)
            decoder = SSEDecoder()
            total_bytes = 0
            record["dispatch_ns"] = time.perf_counter_ns()
            try:
                async with asyncio.timeout(timeout_s):
                    async with client.stream(
                        "POST", base_url.rstrip("/") + "/v1/completions",
                        content=payloads[ordinal],
                        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
                        timeout=httpx.Timeout(timeout_s, connect=min(5, timeout_s)),
                    ) as response:
                        record["headers_ns"] = time.perf_counter_ns()
                        record["http_status"] = response.status_code
                        if response.status_code != 200:
                            raise StreamError(f"HTTP {response.status_code}")
                        if response.headers.get("content-type", "").split(";")[0].strip() != "text/event-stream":
                            raise StreamError("Response is not an SSE stream")
                        async for chunk in response.aiter_bytes():
                            received = time.perf_counter_ns()
                            total_bytes += len(chunk)
                            if total_bytes > 8_388_608:
                                raise StreamError("Response exceeds the trace size limit")
                            for data in decoder.feed(chunk):
                                trace.accept(data, received)
                        # Consume HTTP framing through EOF so the connection can
                        # return to the pool. E2E still ends at the DONE receipt.
                        for data in decoder.feed(b"", final=True):
                            trace.accept(data, time.perf_counter_ns())
                        trace.validate(exact_output)
                record["status"] = "success"
            except asyncio.CancelledError:
                record.update(
                    status="cancelled", error_category="CancelledError",
                    error_message="Cell cancelled", error_ns=time.perf_counter_ns(),
                )
                stop.set()
                raise
            except Exception as exc:
                record.update(
                    status="failed", error_category=type(exc).__name__,
                    error_message=str(exc)[:1000], error_ns=time.perf_counter_ns(),
                )
                stop.set()
            finally:
                record["response_text"] = "".join(trace.text)
                records.append(record)
                events.extend(trace.events)

    sampler = asyncio.create_task(sample_lag())
    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    try:
        await asyncio.gather(*workers)
    finally:
        for task in workers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*workers, return_exceptions=True)
        sampling_done.set()
        sampler.cancel()
        await asyncio.gather(sampler, return_exceptions=True)
    records.sort(key=lambda row: row["sequence_ordinal"])
    from .metrics import summarize
    return {
        "records": records, "events": events,
        # A probe that honors EOS is not a fixed-length measurement.
        "summary": summarize(records, expected_requests=len(prompts)) if exact_output else None,
        "client_loop_lag_ms": loop_lag_ms, "client_lag_interval_ms": 10,
    }
