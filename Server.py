import os
import psycopg2
from psycopg2.extras import Json
import numpy as np
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, iirnotch

app = FastAPI()

# =====================
# Configuración CORS
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://neuronatech.vercel.app",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# Conexión a PostgreSQL
# =====================
DB_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

# =====================
# Parámetros señal
# =====================
FS = 250  # Hz

def bandpass_filter(data, low=1, high=40, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, data)

def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    nyq = 0.5 * fs
    b, a = iirnotch(f0/nyq, Q)
    return filtfilt(b, a, data)

def apply_filters(values):
    arr = np.array(values, dtype=np.float32)
    arr = notch_filter(arr, f0=50.0)
    arr = bandpass_filter(arr, 1, 40)
    return arr.tolist()

def compute_fft(values, fs=FS):
    arr = np.array(values, dtype=np.float32)
    freqs = np.fft.rfftfreq(len(arr), d=1/fs)
    fft_vals = np.abs(np.fft.rfft(arr))
    return {"freqs": freqs.tolist(), "fft": fft_vals.tolist()}

# =====================
# Rutas API
# =====================
@app.get("/")
async def root():
    return {"message": "Servidor funcionando ✅"}

@app.get("/signals")
async def get_signals(limit: int = Query(2500, ge=1, le=10000)):
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": r[3]}
        for r in rows
    ]

@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(2500, ge=1, le=10000)):
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals_processed ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": r[3]}
        for r in rows
    ]

@app.get("/signals/fft")
async def get_signals_fft(limit: int = Query(10, ge=1, le=100)):
    cursor.execute(
        "SELECT id, timestamp, device_id, fft FROM brain_signals_fft ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "fft": r[3]}
        for r in rows
    ]

# =====================
# WebSocket
# =====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 Cliente conectado al WebSocket")

    try:
        while True:
            data_bytes = await websocket.receive_bytes()
            print(f"📩 Paquete binario recibido: {len(data_bytes)} bytes")

            try:
                values = np.frombuffer(data_bytes, dtype=np.float32).tolist()
            except Exception as e:
                print("❌ Error al decodificar paquete:", e)
                continue

            # Guardar señal cruda
            for v in values:
                cursor.execute(
                    "INSERT INTO brain_signals (device_id, value_uv) VALUES (%s, %s)",
                    ("pcb_001", v),
                )

            # Filtrar y calcular FFT
            try:
                filtered = apply_filters(values)
                fft_data = compute_fft(filtered)

                # Guardar señal filtrada
                for fv in filtered:
                    cursor.execute(
                        "INSERT INTO brain_signals_processed (device_id, value_uv) VALUES (%s, %s)",
                        ("pcb_001", fv),
                    )

                # Guardar FFT en tabla separada
                cursor.execute(
                    "INSERT INTO brain_signals_fft (device_id, fft) VALUES (%s, %s)",
                    ("pcb_001", Json(fft_data)),
                )

                conn.commit()
                print(f"✅ Guardados {len(values)} crudos, {len(filtered)} filtrados y FFT")
            except Exception as e:
                conn.rollback()
                print("⚠️ Error en filtrado/FFT:", e)

            await websocket.send_text(
                f"Guardados {len(values)} crudos, {len(values)} filtrados y FFT"
            )
    except Exception as e:
        print("⚠️ Error en WebSocket:", e)
    finally:
        await websocket.close()
        print("❌ Cliente desconectado")
