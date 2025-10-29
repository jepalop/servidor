import os
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, iirnotch

# ============================================================
# FASTAPI SETUP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # para Render y localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE
# ============================================================
DB_URL = os.getenv("DATABASE_URL")

def connect_db():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn, conn.cursor()

conn, cursor = connect_db()

def get_cursor():
    """Verifica que la conexión con PostgreSQL siga activa."""
    global conn, cursor
    try:
        cursor.execute("SELECT 1;")
    except (Exception, psycopg2.Error):
        print("🔄 Reconnecting to PostgreSQL...")
        conn, cursor = connect_db()
    return cursor

# ============================================================
# FILTROS DE SEÑAL
# ============================================================
FS = 250  # Frecuencia de muestreo (Hz)

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

# ============================================================
# ENDPOINT RAÍZ
# ============================================================
@app.get("/")
async def root():
    return {"message": "🧠 Servidor activo y listo"}

# ============================================================
# WEBSOCKET — RECEPCIÓN DE DATOS BINARIOS
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 App Android conectada al WebSocket")

    try:
        while True:
            # Esperar bloque binario desde la app
            data = await websocket.receive_bytes()
            try:
                # Esperamos flotantes little-endian: [front, ref, front, ref, ...]
                floats = np.frombuffer(data, dtype="<f4")
                if len(floats) % 2 != 0:
                    print("⚠️ Longitud de paquete inválida, ignorando")
                    continue

                # Separar canales
                pairs = floats.reshape(-1, 2)
                front = pairs[:, 0]
                ref = pairs[:, 1]

                # 🧮 Re-referenciado
                reref = front - ref

                # 🔹 Filtrado
                filtered = apply_filters(reref)

                # Generar timestamps uniformes
                timestamps = [
                    datetime.now() + timedelta(seconds=i / FS)
                    for i in range(len(filtered))
                ]

                # Insertar en la base de datos
                c = get_cursor()
                c.executemany(
                    """
                    INSERT INTO brain_signals_processed (device_id, timestamp, value_uv)
                    VALUES (%s, %s, %s)
                    """,
                    [("reref", ts, float(v)) for ts, v in zip(timestamps, filtered)],
                )
                conn.commit()

                print(f"✅ {len(filtered)} muestras re-referenciadas guardadas")
                await websocket.send_text(f"OK:{len(filtered)}")

            except Exception as e:
                conn.rollback()
                print("⚠️ Error procesando paquete:", e)
                await websocket.send_text(f"ERROR:{e}")

    except Exception as e:
        print("⚠️ Error en WebSocket principal:", e)

    finally:
        print("❌ App desconectada del WebSocket")

# ============================================================
# ENDPOINT REST — CONSULTA PARA EL FRONTEND
# ============================================================
@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(750, ge=1, le=10000)):
    """
    Devuelve las últimas muestras procesadas almacenadas
    (re-referenciadas y filtradas) para el frontend.
    """
    c = get_cursor()
    c.execute(
        """
        SELECT id, timestamp, device_id, value_uv
        FROM brain_signals_processed
        ORDER BY id DESC
        LIMIT %s;
        """,
        (limit,),
    )
    rows = c.fetchall()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "device_id": r[2],
            "value_uv": float(r[3]),
        }
        for r in rows
    ]
