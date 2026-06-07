import streamlit as st
import json
import time
import asyncio
import aiohttp
import matplotlib.pyplot as plt

STREAM_URL = "http://127.0.0.1:8000/stream"
START_URL = "http://127.0.0.1:8000/start-processing"


# -----------------------------------
# Streamlit Page Setup
# -----------------------------------
st.set_page_config(page_title="UPI Fraud Stream", layout="wide")
st.title("⚡ Real-Time UPI Transaction Fraud Monitor")


# -----------------------------------
# UI Layout
# -----------------------------------
col1, col2 = st.columns([3, 2])

col1.subheader("📟 Live UPI Transactions (LEFT Console)")
left_console = col1.empty()

col2.subheader("🚨 Fraud Detected (RIGHT Console)")
right_console = col2.empty()

stats_box = st.empty()
latency_chart_box = st.empty()   # NEW GRAPH AREA


# -----------------------------------
# Initialize Session State
# -----------------------------------
if "events_all" not in st.session_state:
    st.session_state.events_all = []

if "fraud_events" not in st.session_state:
    st.session_state.fraud_events = []

if "rows_processed" not in st.session_state:
    st.session_state.rows_processed = 0

if "fraud_count" not in st.session_state:
    st.session_state.fraud_count = 0

if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()

if "processing_started" not in st.session_state:
    st.session_state.processing_started = False

# NEW --- LATENCY TRACKING
if "checking_ts" not in st.session_state:
    st.session_state.checking_ts = {}

if "latencies" not in st.session_state:
    st.session_state.latencies = []   # store latency in ms


# -----------------------------------
# Auto-Start Backend Processing
# -----------------------------------
import requests
if not st.session_state.processing_started:
    try:
        requests.post(
            START_URL,
            json={"csv_path": "simulation_dataset_1M.csv", "tps": 1000, "loop": False},
            timeout=3
        )
        st.session_state.processing_started = True
        st.success("Backend processing started!")
    except:
        st.warning("Backend not reachable.")


# -----------------------------------
# SSE STREAM READER (NO THREADS)
# -----------------------------------
async def sse_stream():
    async with aiohttp.ClientSession() as session:
        async with session.get(STREAM_URL) as resp:

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line.startswith("data:"):
                    continue

                payload = line[5:].strip()
                st.session_state.events_all.append(payload)

                # Parse JSON
                try:
                    evt = json.loads(payload)
                except:
                    continue

                row_id = evt.get("row_id")
                status = evt.get("status")
                ts = time.time()

                st.session_state.rows_processed += 1

                # ---------------- LATENCY CAPTURE ----------------
                if status == "checking":
                    st.session_state.checking_ts[row_id] = ts

                if status in ("ok", "fraud"):
                    if row_id in st.session_state.checking_ts:
                        latency = (ts - st.session_state.checking_ts.pop(row_id)) * 1000
                        st.session_state.latencies.append(latency)
                # -------------------------------------------------

                if status == "fraud":
                    st.session_state.fraud_count += 1
                    st.session_state.fraud_events.append(payload)

                # LEFT console
                left_console.markdown(
                    "```\n" +
                    "\n".join(st.session_state.events_all[-40:]) +
                    "\n```"
                )

                # RIGHT console
                right_console.markdown(
                    "```\n" +
                    "\n".join(st.session_state.fraud_events[-40:]) +
                    "\n```"
                )

                # STATS
                elapsed = ts - st.session_state.start_time
                tps_live = st.session_state.rows_processed / max(elapsed, 0.001)

                stats_box.markdown(f"""
                ### 📊 Live Stats
                - Processed: {st.session_state.rows_processed}
                - Frauds: {st.session_state.fraud_count}
                - Throughput: {tps_live:.1f} tx/sec
                - Avg Latency: { (sum(st.session_state.latencies)/len(st.session_state.latencies)) if st.session_state.latencies else 0 :.2f} ms
                """)

                # ---------------- LATENCY GRAPH ----------------
                # if len(st.session_state.latencies) > 2:
                #     plt.figure(figsize=(7,3))
                #     plt.plot(st.session_state.latencies[-200:], marker='.', linewidth=1)
                #     plt.title("Latency Over Time (last 200 events)")
                #     plt.xlabel("Event Index")
                #     plt.ylabel("Latency (ms)")
                #     plt.grid(True)
                #     latency_chart_box.pyplot(plt)
                #     plt.close()
                # ------------------------------------------------


# -----------------------------------
# Run SSE Stream inside Streamlit
# -----------------------------------
asyncio.run(sse_stream())
