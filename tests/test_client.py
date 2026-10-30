import asyncio
import json
import unittest
from contextlib import asynccontextmanager

from inference_bench.client import SSEDecoder, StreamError, run_cell
from inference_bench.provenance import digest_json


def prompt(i=0):
    ids = [2, 123, 456]
    return {"request_id": f"test-{i}", "prompt_ids": ids, "prompt_sha256": digest_json(ids)}


def event(value):
    value = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ("data: " + value + "\r\n\r\n").encode("utf-8")


@asynccontextmanager
async def server(mode="normal"):
    state = {"calls": 0, "active": 0, "maximum": 0, "bodies": []}
    tasks = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        active = False
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode("ascii")
            length = next(
                int(line.split(":", 1)[1]) for line in headers.split("\r\n")
                if line.lower().startswith("content-length:")
            )
            body = json.loads(await reader.readexactly(length))
            state["calls"] += 1
            state["bodies"].append(body)
            state["active"] += 1
            active = True
            state["maximum"] = max(state["maximum"], state["active"])
            if mode == "http-error":
                writer.write(b"HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\n\r\n")
                await writer.drain()
                return
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            if mode == "trickle":
                while True:
                    writer.write(b": keepalive\r\n\r\n")
                    await writer.drain()
                    await asyncio.sleep(0.01)
            await asyncio.sleep(0.015)
            data = [
                event({"choices": [{"index": 0, "text": "", "finish_reason": None}]}),
                event({"choices": [{"index": 0, "text": "hé", "finish_reason": None}]}),
                event({"choices": [{"index": 0, "text": "llo", "finish_reason": "length"}]}),
                event({"choices": [], "usage": {
                    "prompt_tokens": len(body["prompt"]) + (mode == "bad-usage"),
                    "completion_tokens": body["max_tokens"],
                }}),
            ]
            if mode != "missing-done":
                data.append(event("[DONE]"))
            payload = b"".join(data)
            if mode == "fragmented":
                for byte in payload:
                    writer.write(bytes([byte]))
                    await writer.drain()
                    await asyncio.sleep(0)
            else:
                writer.write(payload)
                await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass
        finally:
            if active:
                state["active"] -= 1
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass
            tasks.discard(task)

    listener = await asyncio.start_server(handle, "127.0.0.1", 0)
    origin = f"http://127.0.0.1:{listener.sockets[0].getsockname()[1]}"
    try:
        yield origin, state
    finally:
        listener.close()
        await listener.wait_closed()
        remaining = list(tasks)
        for task in remaining:
            task.cancel()
        await asyncio.gather(*remaining, return_exceptions=True)


class DecoderTests(unittest.TestCase):
    def test_utf8_fragmentation_comments_multiline_and_crlf(self):
        wire = ': comment\r\ndata: {"text":\r\ndata: "hé"}\r\n\r\ndata: [DONE]\n\n'.encode()
        decoder = SSEDecoder()
        result = []
        for byte in wire:
            result.extend(decoder.feed(bytes([byte])))
        result.extend(decoder.feed(b"", final=True))
        self.assertEqual(result, ['{"text":\n"hé"}', "[DONE]"])

    def test_truncation_and_unbounded_events(self):
        decoder = SSEDecoder()
        decoder.feed(b'data: {"x": 1}')
        with self.assertRaises(StreamError):
            decoder.feed(b"", final=True)
        with self.assertRaises(StreamError):
            SSEDecoder(max_event_chars=8).feed(b"data: too much text\n\n")

    def test_multiple_events_in_one_large_read(self):
        decoder = SSEDecoder(max_event_chars=20)
        self.assertEqual(decoder.feed(b"data: ok\n\n" * 10), ["ok"] * 10)


class HTTPTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_concurrency_and_token_usage(self):
        async with server() as (origin, state):
            result = await run_cell(
                origin, [prompt(i) for i in range(8)], concurrency=4,
                output_tokens=3, seed=10, timeout_s=2,
            )
        self.assertTrue(result["summary"]["valid"])
        self.assertEqual(state["maximum"], 4)
        self.assertEqual(state["calls"], 8)
        self.assertEqual(result["summary"]["successful_output_tokens"], 24)
        self.assertEqual([r["response_text"] for r in result["records"]], ["héllo"] * 8)
        self.assertEqual(sorted(b["seed"] for b in state["bodies"]), list(range(10, 18)))
        self.assertTrue(all(b["ignore_eos"] for b in state["bodies"]))
        self.assertTrue(all(r["content_event_count"] == 2 for r in result["records"]))

    async def test_fragmented_unicode_and_terminal_usage(self):
        async with server("fragmented") as (origin, _):
            result = await run_cell(origin, [prompt()], concurrency=1, output_tokens=3, seed=0)
        row = result["records"][0]
        self.assertEqual(row["status"], "success")
        self.assertEqual(row["response_text"], "héllo")
        self.assertLessEqual(row["headers_ns"], row["first_content_ns"])
        self.assertLessEqual(row["last_content_ns"], row["done_ns"])

    async def test_failure_stops_new_dispatch_and_is_not_retried(self):
        for mode in ("bad-usage", "missing-done", "http-error"):
            with self.subTest(mode=mode):
                async with server(mode) as (origin, state):
                    result = await run_cell(
                        origin, [prompt(i) for i in range(8)], concurrency=1,
                        output_tokens=3, seed=0, timeout_s=2,
                    )
                self.assertFalse(result["summary"]["valid"])
                self.assertEqual(state["calls"], 1)
                self.assertEqual(result["summary"]["failed_requests"], 1)
                self.assertEqual(result["summary"]["unattempted_requests"], 7)

    async def test_absolute_timeout_applies_to_trickling_stream(self):
        async with server("trickle") as (origin, state):
            result = await run_cell(
                origin, [prompt()], concurrency=1, output_tokens=3, seed=0, timeout_s=0.08,
            )
        self.assertEqual(state["calls"], 1)
        self.assertEqual(result["records"][0]["error_category"], "TimeoutError")
        self.assertFalse(result["summary"]["valid"])

    async def test_remote_endpoint_and_wrong_prompt_hash_are_rejected(self):
        with self.assertRaises(ValueError):
            await run_cell("https://example.com", [prompt()], concurrency=1, output_tokens=3, seed=0)
        bad = {**prompt(), "prompt_sha256": "0" * 64}
        with self.assertRaises(ValueError):
            await run_cell("http://127.0.0.1:1", [bad], concurrency=1, output_tokens=3, seed=0)
