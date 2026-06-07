# app.py
import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Optional: CatBoost for production scoring
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

app = FastAPI(title="Real-time Fraud Streaming Backend (SSE)")

# ----------------------------------------
# Global broadcast queue & client registry
# ----------------------------------------
# We'll keep a single asyncio.Queue where background processor puts events.
# Each SSE connection will read events from its own queue by reading the global queue snapshot.
# To broadcast to connected clients we store client queues.
CLIENT_QUEUES: List[asyncio.Queue] = []
CLIENT_LOCK = asyncio.Lock()

# Processing control
PROCESSING_TASK: Optional[asyncio.Task] = None
PROCESSING_LOCK = asyncio.Lock()
STOP_PROCESSING = False

# Model metadata (if present)
MODEL_PATH = Path("models/fraud_modelN.cbm")
MODEL_META_PATH = Path("models/fraud_model_metadataN.json")
MODEL = None
MODEL_META = {}

# Precompute global median/mad for naive z-score fallback if dataset used directly
GLOBAL_MEDIAN = None
GLOBAL_MAD = None

# default thresholds (fallback)
DEFAULT_THRESHOLDS = {
    "max_f1": 0.85,
    "block": 0.99,
    "review": 0.77,
    "monitor": 0.5
}

# ------------------------
# Pydantic request models
# ------------------------
class StartRequest(BaseModel):
    csv_path: str
    tps: Optional[float] = 1000.0
    loop: Optional[bool] = False
    # optional: override thresholds or model metadata path can be added


# ------------------------
# Utilities
# ------------------------
def iso_ts():
    return datetime.utcnow().isoformat() + "Z"


async def broadcast_event(event: Dict):
    """
    Put event into every connected client's queue.
    """
    data = event.copy()
    data_json = json.dumps(data, default=str)
    # create shallow copy list to avoid concurrency issues
    async with CLIENT_LOCK:
        queues = list(CLIENT_QUEUES)
    for q in queues:
        try:
            # put_nowait to avoid blocking broadcast; if queue full drop oldest
            q.put_nowait(data_json)
        except asyncio.QueueFull:
            try:
                _ = q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(data_json)
            except Exception:
                pass


def safe_load_model():
    global MODEL, MODEL_META
    if MODEL_PATH.exists() and CATBOOST_AVAILABLE:
        try:
            m = CatBoostClassifier()
            m.load_model(str(MODEL_PATH))
            MODEL = m
            if MODEL_META_PATH.exists():
                with open(MODEL_META_PATH, "r") as f:
                    MODEL_META = json.load(f)
            else:
                MODEL_META = {}
            app.logger = getattr(app, "logger", None)
            return True
        except Exception:
            MODEL = None
            MODEL_META = {}
            return False
    return False


def load_global_stats(csv_path: str):
    """
    Compute global median and mad of transaction_amount from CSV
    Useful for quick amount_zscore fallback.
    """
    global GLOBAL_MEDIAN, GLOBAL_MAD
    try:
        df = pd.read_csv(csv_path, usecols=["transaction_amount"], nrows=200_000)
        GLOBAL_MEDIAN = float(df["transaction_amount"].median())
        mad = float((df["transaction_amount"] - GLOBAL_MEDIAN).abs().median())
        GLOBAL_MAD = max(mad, 1e-6)
    except Exception:
        GLOBAL_MEDIAN = 0.0
        GLOBAL_MAD = 1.0


def extract_sender_bank_code(sender_upi: str):
    """
    lightweight mapping: try to parse bank substring from UPI (e.g., user@oksbi -> oksbi)
    map to small integer codes for model features.
    """
    if not isinstance(sender_upi, str):
        return 0
    domain = sender_upi.split("@")[-1].lower()
    # simple mapping - extendable
    BANK_MAP = {
        "oksbi": 1, "okicici": 2, "okhdfcbank": 3, "okaxis": 4, "okybl": 5,
        "oksbi.in": 1, "okicici.in": 2
    }
    return BANK_MAP.get(domain, 0)


def compute_amount_zscore(amount: float):
    # fallback using global median/mad
    if GLOBAL_MAD and GLOBAL_MAD > 0:
        return (amount - GLOBAL_MEDIAN) / GLOBAL_MAD
    return 0.0


def build_feature_vector_from_row(row: Dict) -> List[float]:
    """
    Attempt to build the 14-feature vector expected by the model from the raw row.
    This implementation is defensive: tries to use provided fields; otherwise uses fallbacks.
    Order must match training FEATURES order.
    """
    # fields expected:
    # transaction_amount, amount_zscore, hour_of_day, is_night, day_of_week,
    # device_change_flag, location_risk_level, location_velocity_flag,
    # sender_bank_code, receiver_bank_code, velocity_1h, failed_attempts_24h,
    # fraud_history_flag, is_new_beneficiary

    amount = float(row.get("transaction_amount") or row.get("amount") or 0.0)
    amount_zscore = float(row.get("amount_zscore") or compute_amount_zscore(amount))
    hour_of_day = int(row.get("hour_of_day") or 0)
    is_night = int(row.get("is_night") or (1 if hour_of_day <= 5 else 0))
    day_of_week = int(row.get("day_of_week") or 0)
    device_change_flag = int(row.get("device_change_flag") or 0)
    location_risk_level = int(row.get("location_risk_level") or 0)
    location_velocity_flag = int(row.get("location_velocity_flag") or 0)
    sender_code = int(row.get("sender_bank_code") or extract_sender_bank_code(row.get("sender_upi", "")))
    receiver_code = int(row.get("receiver_bank_code") or extract_sender_bank_code(row.get("receiver_upi", "")))
    velocity_1h = int(row.get("velocity_1h") or 0)
    failed_attempts_24h = int(row.get("failed_attempts_24h") or 0)
    fraud_history_flag = int(row.get("fraud_history_flag") or 0)
    is_new_beneficiary = int(row.get("is_new_beneficiary") or 0)

    vec = [
        amount,
        amount_zscore,
        hour_of_day,
        is_night,
        day_of_week,
        device_change_flag,
        location_risk_level,
        location_velocity_flag,
        sender_code,
        receiver_code,
        velocity_1h,
        failed_attempts_24h,
        fraud_history_flag,
        is_new_beneficiary,
    ]
    return vec


def heuristic_score(row: Dict) -> Dict:
    """
    A safe fallback scoring logic that produces a risk score and a reason.
    Weighted linear combination of features (normalized).
    """
    amount = float(row.get("transaction_amount") or 0.0)
    z = compute_amount_zscore(amount)
    is_night = int(row.get("is_night") or 0)
    device_change = int(row.get("device_change_flag") or 0)
    loc_risk = int(row.get("location_risk_level") or 0)
    loc_vel = int(row.get("location_velocity_flag") or 0)
    velocity = int(row.get("velocity_1h") or 0)
    failed = int(row.get("failed_attempts_24h") or 0)
    fraud_hist = int(row.get("fraud_history_flag") or 0)
    new_ben = int(row.get("is_new_beneficiary") or 0)

    # weights tuned to give small % of fraud from simulation
    score = (
        0.18 * (np.tanh(z / 3.0)) +
        0.55 * is_night +
        1.6 * device_change +
        0.7 * (loc_risk / 3.0) +
        2.5 * loc_vel +
        0.2 * velocity +
        0.5 * failed +
        2.0 * fraud_hist +
        0.9 * new_ben
    )
    prob = 1.0 / (1.0 + np.exp(- (score - 1.6)))  # sigmoid calibrated
    reason = []
    if abs(z) > 5:
        reason.append("amount_outlier")
    if device_change:
        reason.append("device_change")
    if loc_vel:
        reason.append("impossible_travel")
    if failed > 0:
        reason.append("failed_attempts")
    if fraud_hist:
        reason.append("history")
    if new_ben:
        reason.append("new_beneficiary")
    reason = reason or ["score_based"]
    return {"score": float(prob), "reasons": reason}


# ------------------------
# Processing logic
# ------------------------
# ------------------------
# High-throughput processing engine (replace previous process_csv_stream)
# ------------------------
import asyncio
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Tuple

# TUNABLES
QUEUE_MAXSIZE = 20000          # backpressure queue (large enough for bursts)
NUM_WORKERS = 4                # number of worker coroutines (tune to CPU cores)
BATCH_SIZE = 128               # batch size for vectorized inference (tune)
BATCH_TIMEOUT_SEC = 0.05       # flush batch after 50 ms if not full
THREADPOOL_SIZE = 2            # for any blocking CPU ops (optional)

# We'll keep a small ThreadPoolExecutor to offload any blocking code if needed
_threadpool = ThreadPoolExecutor(max_workers=THREADPOOL_SIZE)


async def _worker_loop(row_queue: asyncio.Queue, worker_id: int, thresholds: Dict):
    """
    Worker: collects rows into a batch, vectorizes features, runs model (or heuristic),
    and emits final events per row. Runs forever until cancelled.
    Each queued item is a tuple (row_id, row_dict, ds_preview).
    """
    loop = asyncio.get_event_loop()
    batch_items = []
    last_flush = perf_counter()

    while True:
        try:
            # wait for first item with a short timeout so we can periodically flush
            try:
                item = await asyncio.wait_for(row_queue.get(), timeout=BATCH_TIMEOUT_SEC)
                batch_items.append(item)
            except asyncio.TimeoutError:
                # no new item in timeout - proceed to flush if batch non-empty
                pass

            # Keep pulling without waiting until we have BATCH_SIZE or timeout
            while len(batch_items) < BATCH_SIZE:
                try:
                    item = row_queue.get_nowait()
                    batch_items.append(item)
                except asyncio.QueueEmpty:
                    break

            # If we have a batch, process it
            if batch_items:
                # Build vectorized feature matrix for model predict
                rows_ids, rows_dicts, previews = zip(*batch_items)
                # Vectorize feature build (fast pure-Python loops; can be optimized further)
                mats = [build_feature_vector_from_row(r) for r in rows_dicts]  # list of lists
                import numpy as _np
                X = _np.array(mats, dtype=float)

                # Run model inference — run inside threadpool if predict_proba is blocking
                risk_probs = None
                if MODEL is not None:
                    try:
                        # CatBoost predict_proba accepts numpy arrays
                        # if predict_proba is CPU-bound C code, this is fast; otherwise run in threadpool
                        # We'll try direct call and fallback to threadpool on unexpected blocking exceptions
                        risk_probs = MODEL.predict_proba(X)[:, 1]
                    except Exception:
                        # fallback to threadpool to avoid blocking event loop
                        risk_probs = await loop.run_in_executor(_threadpool, lambda: MODEL.predict_proba(X)[:, 1])
                else:
                    # heuristic scoring vectorized
                    # fallback: call heuristic per-row (can be optimized)
                    risk_probs = []
                    for r in rows_dicts:
                        sc = heuristic_score(r)
                        risk_probs.append(sc["score"])
                    risk_probs = _np.array(risk_probs, dtype=float)

                # Emit events per row immediately after scoring (order preserved within batch)
                for i, rid in enumerate(rows_ids):
                    prob = float(risk_probs[i])
                    # final decision uses review threshold by default (same heuristic as earlier)
                    if prob >= thresholds.get("review", DEFAULT_THRESHOLDS["review"]):
                        status = "fraud"
                        reason = ["model_score"]
                    else:
                        status = "ok"
                        reason = []
                    final_event = {
                        "row_id": int(rid),
                        "status": status,
                        "dataset_row": previews[i],
                        "details": f"fraud_score={prob:.4f}; reasons={','.join(reason)}" if status == "fraud" else f"fraud_score={prob:.4f}",
                        "timestamp": iso_ts(),
                        "fraud_score": prob,
                        "reason": reason,
                        "fraud_type": None
                    }
                    # broadcast (non-blocking)
                    await broadcast_event(final_event)

                # done processing this batch: mark tasks done
                for _ in range(len(batch_items)):
                    row_queue.task_done()

                # reset batch
                batch_items = []
                last_flush = perf_counter()

        except asyncio.CancelledError:
            # gracefully exit
            break
        except Exception as e:
            # log and continue
            # emit a single error event for debugging
            await broadcast_event({
                "row_id": -1,
                "status": "error",
                "dataset_row": "",
                "details": f"worker_exception: {str(e)}",
                "timestamp": iso_ts()
            })
            # small sleep to avoid tight error loop
            await asyncio.sleep(0.01)


async def process_csv_stream_high_throughput(csv_path: str, tps: float = 1000.0, loop: bool = False):
    """
    New high-throughput processing function.
    - Emits 'checking' event immediately for each row
    - Enqueues rows into an asyncio.Queue
    - Spins up NUM_WORKERS workers that batch and perform inference
    """
    global STOP_PROCESSING

    if not Path(csv_path).exists():
        await broadcast_event({
            "row_id": -1,
            "status": "error",
            "dataset_row": f"CSV not found: {csv_path}",
            "details": "",
            "timestamp": iso_ts()
        })
        return

    # precompute global stats for zscore fallback
    load_global_stats(csv_path)

    # Ensure model loaded if present
    safe_load_model()
    thresholds = MODEL_META.get("thresholds", DEFAULT_THRESHOLDS) if MODEL_META else DEFAULT_THRESHOLDS

    # create queue and start workers
    row_queue = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
    workers = []
    for wid in range(NUM_WORKERS):
        w = asyncio.create_task(_worker_loop(row_queue, wid, thresholds))
        workers.append(w)

    # Producer: read CSV in chunks and enqueue
    row_id = 0
    try:
        while True:
            STOP_PROCESSING = False
            # stream by chunks to avoid loading all
            with open(csv_path, newline="", encoding="utf-8") as fh:
                for chunk in pd.read_csv(fh, chunksize=10000):
                    for _, r in chunk.iterrows():
                        if STOP_PROCESSING:
                            raise asyncio.CancelledError()

                        row_id += 1
                        # prepare ds_preview for UI
                        ds_preview = {
                            "transaction_id": r.get("transaction_id") if "transaction_id" in r else f"row_{row_id}",
                            "sender_upi": r.get("sender_upi", ""),
                            "receiver_upi": r.get("receiver_upi", ""),
                            "amount": float(r.get("transaction_amount") or r.get("amount") or 0.0)
                        }

                        # Emit checking event immediately (non-blocking)
                        checking_event = {
                            "row_id": row_id,
                            "status": "checking",
                            "dataset_row": ds_preview,
                            "details": "",
                            "timestamp": iso_ts()
                        }
                        await broadcast_event(checking_event)

                        # enqueue the work item; if queue is full, this await will apply backpressure
                        await row_queue.put((row_id, r.to_dict(), ds_preview))

                        # throttle producer to target TPS
                        if tps and tps > 0:
                            await asyncio.sleep(1.0 / float(tps))

                    # continue next chunk
            if not loop:
                break
            # else start next loop of CSV
    except asyncio.CancelledError:
        # signal workers to stop by cancelling them
        STOP_PROCESSING = True
    finally:
        # wait until queue is drained
        await row_queue.join()
        # cancel workers
        for w in workers:
            w.cancel()
        # gather to ensure cancellation
        await asyncio.gather(*workers, return_exceptions=True)

    # broadcast finished
    await broadcast_event({
        "row_id": -1,
        "status": "finished",
        "dataset_row": f"Completed processing: {csv_path}",
        "details": "",
        "timestamp": iso_ts()
    })


# ------------------------
# Endpoints
# ------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": MODEL is not None}

@app.get("/metadata")
async def metadata():
    meta = {"model_loaded": MODEL is not None}
    if MODEL_META:
        meta.update(MODEL_META)
    else:
        meta["thresholds"] = DEFAULT_THRESHOLDS
    return meta

@app.post("/start-processing")
async def start_processing(req: StartRequest, background_tasks: BackgroundTasks):
    global PROCESSING_TASK, STOP_PROCESSING
    if not Path(req.csv_path).exists():
        raise HTTPException(status_code=404, detail="CSV path not found")

    async with PROCESSING_LOCK:
        if PROCESSING_TASK is not None and not PROCESSING_TASK.done():
            raise HTTPException(status_code=409, detail="Processing already running")
        # start background processing task
        PROCESSING_TASK = asyncio.create_task(process_csv_stream_high_throughput(req.csv_path, tps=req.tps, loop=req.loop))
    return {"status": "started", "csv_path": req.csv_path, "tps": req.tps, "loop": req.loop}

@app.post("/stop-processing")
async def stop_processing():
    global STOP_PROCESSING, PROCESSING_TASK
    STOP_PROCESSING = True
    # attempt to cancel
    if PROCESSING_TASK:
        try:
            PROCESSING_TASK.cancel()
        except Exception:
            pass
    return {"status": "stopping"}

@app.get("/stream")
async def stream(request: Request):
    """
    SSE endpoint returning streaming JSON events. Each event data is a JSON string.
    Clients should keep connection open and process each 'data' message as JSON.
    """
    client_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)

    # Register client
    async with CLIENT_LOCK:
        CLIENT_QUEUES.append(client_queue)

    async def event_generator():
        try:
            # while client connected:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    # wait for next event (timeout to let us detect disconnects)
                    data_json = await asyncio.wait_for(client_queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    # send a keepalive comment to keep the SSE open
                    yield {"event": "keepalive", "data": json.dumps({"ts": iso_ts()})}
                    continue
                # send actual event
                yield {"event": "message", "data": data_json}
        finally:
            # cleanup
            async with CLIENT_LOCK:
                try:
                    CLIENT_QUEUES.remove(client_queue)
                except ValueError:
                    pass

    return EventSourceResponse(event_generator())

# Root
@app.get("/")
async def root():
    return {"message": "Fraud streaming backend. Use /stream (SSE) and /start-processing to begin."}
