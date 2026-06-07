# sse_monitor.py
import json
import time
import asyncio
import statistics
import websockets    # pip install websockets sseclient-py
import sys
import ssl
from urllib.parse import urlparse
import aiohttp       # pip install aiohttp

STREAM_URL = "http://127.0.0.1:8000/stream"

# This client uses aiohttp to connect to SSE and parses events
import aiohttp

async def monitor_sse(url):
    checking_ts = {}   # row_id -> timestamp when checking seen
    latencies = []     # list of (row_id, latency_ms)
    fraud_count = 0
    total_final = 0
    last_report = time.time()
    session = aiohttp.ClientSession()
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                print("SSE connection failed", resp.status)
                return
            print("Connected to SSE. Listening...")
            async for line in resp.content:
                try:
                    text = line.decode().strip()
                except Exception:
                    continue
                if not text:
                    continue
                # SSE messages have lines like "data: <json>"
                if text.startswith("data:"):
                    payload = text[len("data:"):].strip()
                    try:
                        ev = json.loads(payload)
                    except Exception:
                        # might be keepalive or other event
                        continue
                    row_id = ev.get("row_id")
                    status = ev.get("status")
                    ts = time.time()
                    # record checking timestamp
                    if status == "checking":
                        checking_ts[row_id] = ts
                    elif status in ("ok", "fraud"):
                        total_final += 1
                        if row_id in checking_ts:
                            latency_ms = (ts - checking_ts.pop(row_id)) * 1000.0
                            latencies.append(latency_ms)
                        else:
                            # no prior checking seen (possible if we started monitor late)
                            pass
                        if status == "fraud":
                            fraud_count += 1
                # periodic reporting
                if time.time() - last_report > 5:
                    last_report = time.time()
                    n = len(latencies)
                    if n:
                        avg = statistics.mean(latencies)
                        p50 = statistics.median(latencies)
                        p95 = sorted(latencies)[int(0.95 * n) - 1] if n >= 20 else max(latencies)
                        print(f"[stats] samples={n} total_final={total_final} frauds={fraud_count} avg={avg:.1f}ms p50={p50:.1f}ms p95={p95:.1f}ms queue_unmatched={len(checking_ts)}")
                    else:
                        print(f"[stats] samples=0 total_final={total_final} frauds={fraud_count} queue_unmatched={len(checking_ts)}")
    except asyncio.CancelledError:
        pass
    finally:
        await session.close()



if __name__ == "__main__":
    try:
        asyncio.run(monitor_sse(STREAM_URL))
    except KeyboardInterrupt:
        print("Monitor stopped")
