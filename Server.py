import os
import psycopg2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, iirnotch
import asyncio

# ============================================================
# CONFIGURACIÓN BÁSICA FASTAPI + CORS
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# CONEXIÓN A BASE DE DATOS
# ============================================================
DB_URL = os.getenv("DATABASE_URL")
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

# ============================================================
# PARÁMETROS DE PROCESAMIENTO DE SEÑAL
# ============================================================
FS = 250           # Hz
HPF_HZ = 2
LPF_HZ = 70.0
NOTCH_HZ = 50.0

# ============================================================
# FUNCIONES DE FILTRADO
# ============================================================
def bandpass_filter(data, low=HPF_HZ, high=LPF_HZ, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, data)

def notch_filter(data, f0=NOTCH_HZ, Q=30.0, fs=FS):
    nyq = 0.5 * fs
    b, a = iirnotch(f0 / nyq, Q)
    return filtfilt(b, a, data)

# ============================================================
# FUNCIONES DE RE-REFERENCIADO ROBUSTO
# ============================================================
def mad(x):
    """Median Absolute Deviation (robusto a outliers)."""
    m = np.median(x)
    return np.median(np.abs(x - m)) + 1e-12

def estimate_ab_robust(x, r, fs, trim_s=1.0, k=6.0):
    """Estima a,b en X ≈ a·R + b usando mínimos cuadrados robustos."""
    i0 = int(trim_s * fs)
    i1 = len(x) - i0 if len(x) > 2 * i0 else len(x)
    x1 = x[i0:i1].copy()
    r1 = r[i0:i1].copy()
    x1c = x1 - np.median(x1)
    r1c = r1 - np.median(r1)
    mask = (np.abs(x1c) < k * mad(x1c)) & (np.abs(r1c) < k * mad(r1c))
    if np.sum(mask) < 10:
        mask = np.ones_like(x1c, dtype=bool)
    Xmat = np.column_stack([r1c[mask], np.ones(np.sum(mask))])
    a, b = np.linalg.lstsq(Xmat, x1c[mask], rcond=None)[0]
    return float(a), float(b)

# ============================================================
# ENDPOINTS DE API
# ============================================================
@app.get("/")
async def root():
    return {"message": "Servidor funcionando ✅"}

@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(7500, ge=1, le=10000)):
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals_processed ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": float(r[3])}
        for r in rows
    ]

# ============================================================
# WEBSOCKETS
# ============================================================
clients = set()

@app.websocket("/ws")  # conexión desde app Android
async def websocket_pcb(websocket: WebSocket):
    await websocket.accept()
    print("📡 PCB conectada al WebSocket")

    try:
        while True:
            data_bytes = await websocket.receive_bytes()
            print(f"📩 {datetime.now()} - Paquete recibido: {len(data_bytes)} bytes")

            try:
                # 🔹 Decodificar pares de canales (CH1, CH2)
                data = np.frombuffer(data_bytes, dtype="<f4").reshape(-1, 2)
                ch1 = data[:, 0]
                ch2 = data[:, 1]
            except Exception as e:
                print("❌ Error al decodificar:", e)
                continue

            try:
                # 🔹 Filtrado notch + banda
                ch1_bp = bandpass_filter(notch_filter(ch1))
                ch2_bp = bandpass_filter(notch_filter(ch2))

                # 🔹 Re-referenciado robusto
                a_r, b_r = estimate_ab_robust(ch1_bp, ch2_bp, fs=FS, trim_s=1.0, k=6.0)
                y_rr = ch1_bp - a_r * ch2_bp - b_r
                print(f"🔹 Re-referenciado robusto: a={a_r:.6f}, b={b_r:.6f}")

                # 🔹 Guardar en base de datos (solo señal re-referenciada)
                cursor.executemany(
                    "INSERT INTO brain_signals_processed (device_id, value_uv) VALUES (%s, %s)",
                    [("pcb_001", float(v)) for v in y_rr],
                )
                conn.commit()

                print(f"✅ {datetime.now()} - Guardadas {len(y_rr)} muestras re-referenciadas")

            except Exception as e:
                conn.rollback()
                print("⚠️ Error durante el procesamiento:", e)
                continue

            # Confirmación a la app Android
            await websocket.send_text(
                f"Guardadas {len(y_rr)} re-referenciadas filtradas"
            )

            # 🔹 Reenviar a clientes conectados (frontend)
            dead_clients = []
            for client in clients:
                try:
                    await client.send_bytes(y_rr.astype("<f4").tobytes())
                except:
                    dead_clients.append(client)

            for dc in dead_clients:
                clients.remove(dc)

    except Exception as e:
        print("⚠️ Error WebSocket PCB:", e)
    finally:
        print("❌ PCB desconectada")

@app.websocket("/ws/client")  # frontend -> servidor
async def websocket_client(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("👀 Cliente conectado al WebSocket")

    try:
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print("⚠️ Error WebSocket cliente:", e)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        print("👋 Cliente desconectado")
