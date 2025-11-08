import os
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, sosfiltfilt, iirnotch
from psycopg2.extras import execute_batch

# ============================================================
# FASTAPI SETUP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Render y localhost
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
# FILTROS EEG MEJORADOS
# ============================================================
FS = 250  # Frecuencia de muestreo (Hz)

def butter_filter(lowcut=None, highcut=None, fs=FS, order=4):
    nyq = 0.5 * fs
    if lowcut and highcut:
        sos = butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")
    elif lowcut:
        sos = butter(order, lowcut / nyq, btype="high", output="sos")
    elif highcut:
        sos = butter(order, highcut / nyq, btype="low", output="sos")
    else:
        raise ValueError("Debe especificarse lowcut o highcut")
    return sos

def notch_filter(data, f0=50.0, Q=30.0, fs=FS):
    nyq = 0.5 * fs
    b, a = iirnotch(f0 / nyq, Q)
    return sosfiltfilt(np.array([[b[0], b[1], b[2], 1, a[1], a[2]]]), data)

# Precalcular filtros
SOS_BP = butter_filter(0.5, 70, fs=FS, order=4)

def apply_filters(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = notch_filter(arr, f0=50.0, Q=30.0)
    arr = sosfiltfilt(SOS_BP, arr)
    return arr.tolist()

# ============================================================
# VARIABLES GLOBALES DE SINCRONIZACIÓN TEMPORAL
# ============================================================
start_time = None
sample_counter = 0

# ============================================================
# ENDPOINT RAÍZ
# ============================================================
@app.get("/")
async def root():
    return {"message": "🧠 Servidor activo y listo (sincronización continua habilitada)"}

# ============================================================
# WEBSOCKET — RECEPCIÓN DE DATOS BINARIOS
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global start_time, sample_counter
    await websocket.accept()
    print("📡 App Android conectada al WebSocket")

    try:
        while True:
            data = await websocket.receive_bytes()
            try:
                floats = np.frombuffer(data, dtype="<f4")
                if len(floats) % 2 != 0:
                    print("⚠️ Longitud de paquete inválida, ignorando")
                    continue

                pairs = floats.reshape(-1, 2)
                front = pairs[:, 0]
                ref = pairs[:, 1]

                # 🧮 Re-referenciado
                reref = front - ref

                # 🔹 Filtrado
                filtered = apply_filters(reref)

                # Sincronización temporal continua
                if start_time is None:
                    start_time = datetime.now()
                    sample_counter = 0

                timestamps = [
                    start_time + timedelta(seconds=(sample_counter + i) / FS)
                    for i in range(len(filtered))
                ]
                sample_counter += len(filtered)

                # Inserción eficiente en la base de datos
                c = get_cursor()
                execute_batch(
                    c,
                    """
                    INSERT INTO brain_signals_processed (device_id, timestamp, value_uv)
                    VALUES (%s, %s, %s)
                    """,
                    [("reref", ts, float(v)) for ts, v in zip(timestamps, filtered)],
                    page_size=500,
                )
                conn.commit()

                print(f"✅ {len(filtered)} muestras procesadas y guardadas (t0={timestamps[0]})")
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
    """Devuelve las últimas muestras procesadas para el frontend."""
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
