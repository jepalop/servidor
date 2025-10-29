import os
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, iirnotch
import asyncio

app = FastAPI()

# =====================
# CONFIG CORS
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# DATABASE
# =====================
DB_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DB_URL)
conn.autocommit = False
cursor = conn.cursor()

def get_cursor():
    global conn, cursor
    try:
        cursor.execute("SELECT 1;")
    except:
        conn = psycopg2.connect(DB_URL)
        cursor = conn.cursor()
    return cursor

# =====================
# FILTERS
# =====================
FS = 250

def bandpass_filter(data, low=1, high=40, fs=FS, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    nyq = 0.5 * fs
    b, a = iirnotch(f0 / nyq, Q)
    return filtfilt(b, a, data)

def apply_filters(values):
    arr = np.array(values, dtype=np.float32)
    arr = notch_filter(arr, f0=50.0)
    arr = bandpass_filter(arr, 1, 40)
    return arr.tolist()

# =====================
# WEBSOCKET ENDPOINT
# =====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 App Android conectada al WebSocket")

    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                floats = np.frombuffer(data, dtype="<f4")
                data_pairs = floats.reshape(-1, 2)
                front = data_pairs[:, 0]
                ref = data_pairs[:, 1]
                reref = front - ref

                filtered = apply_filters(reref)
                timestamps = [
                    datetime.now() + timedelta(seconds=i / FS)
                    for i in range(len(filtered))
                ]

                c = get_cursor()
                c.executemany(
                    "INSERT INTO brain_signals_processed (device_id, timestamp, value_uv) VALUES (%s, %s, %s)",
                    [("reref", ts, float(v)) for ts, v in zip(timestamps, filtered)],
                )
                conn.commit()
                print(f"✅ Guardadas {len(filtered)} muestras re-referenciadas")

                await websocket.send_text(f"OK:{len(filtered)}")

            except Exception as e:
                conn.rollback()
                print("⚠️ Error procesando paquete:", e)
                await websocket.send_text(f"ERROR:{e}")

    except Exception as e:
        print("⚠️ Error en WebSocket:", e)
    finally:
        print("❌ App desconectada del WebSocket")
