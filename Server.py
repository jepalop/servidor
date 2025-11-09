import os
import psycopg2
import numpy as np
import struct
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import execute_batch

# ============================================================
# FASTAPI SETUP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE CONNECTION
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
# CONFIGURACIÓN GLOBAL
# ============================================================
FS_DEFAULT = 250  # Hz (valor de referencia)
buffers = {1: [], 2: []}  # 1=front, 2=ref

# ============================================================
# ENDPOINT PRINCIPAL
# ============================================================
@app.get("/")
async def root():
    return {"message": "🧠 Servidor activo y listo (recepción binaria sincronizada habilitada)"}

# ============================================================
# PARSEO DE PAQUETE BINARIO (nuevo formato)
# ============================================================
def parse_binary_packet(packet_bytes):
    """
    Estructura esperada del paquete:
    [device_id:1][timestamp_start:8][sample_rate:2][n_samples:2][floats:4*n]
    """
    header_fmt = "<Bqhh"
    header_size = struct.calcsize(header_fmt)
    if len(packet_bytes) < header_size:
        print("⚠️ Paquete demasiado corto (sin cabecera)")
        return None

    device_id, ts_start, fs, n = struct.unpack_from(header_fmt, packet_bytes, 0)
    if n <= 0 or n > 10000:
        print(f"⚠️ Tamaño inválido n={n}")
        return None

    data_fmt = f"<{n}f"
    expected_size = header_size + struct.calcsize(data_fmt)
    if len(packet_bytes) < expected_size:
        print("⚠️ Paquete incompleto, descartado")
        return None

    samples = struct.unpack_from(data_fmt, packet_bytes, header_size)
    timestamps = [datetime.fromtimestamp(ts_start / 1000.0) + timedelta(seconds=i / fs) for i in range(n)]
    return device_id, timestamps, samples

# ============================================================
# ALINEACIÓN Y RESTA SINCRONIZADA
# ============================================================
def align_and_subtract():
    """
    Busca timestamps coincidentes (o muy cercanos) entre front y ref.
    Calcula front - ref y devuelve [(timestamp, value_uv)].
    """
    if not buffers[1] or not buffers[2]:
        return []

    front_ts, front_vals = zip(*buffers[1])
    ref_ts, ref_vals = zip(*buffers[2])
    ref_ts_np = np.array([ts.timestamp() for ts in ref_ts])
    ref_vals_np = np.array(ref_vals)

    results = []
    for t, v_front in zip(front_ts, front_vals):
        t_sec = t.timestamp()
        idx = np.argmin(np.abs(ref_ts_np - t_sec))
        if abs(ref_ts_np[idx] - t_sec) <= 0.004:  # tolerancia de 4 ms
            diff = v_front - ref_vals_np[idx]
            results.append((t, diff))

    # Mantener últimas 100 muestras en buffers (por posibles retrasos)
    buffers[1] = buffers[1][-100:]
    buffers[2] = buffers[2][-100:]

    return results

# ============================================================
# GUARDADO EN BASE DE DATOS
# ============================================================
def insert_processed_data(rows):
    if not rows:
        return
    try:
        c = get_cursor()
        execute_batch(
            c,
            """
            INSERT INTO brain_signals_processed (device_id, timestamp, value_uv)
            VALUES (%s, %s, %s)
            """,
            [("reref", ts, float(val)) for ts, val in rows],
            page_size=500,
        )
        conn.commit()
        print(f"💾 Guardadas {len(rows)} muestras procesadas")
    except Exception as e:
        conn.rollback()
        print("⚠️ Error al guardar en la BD:", e)

# ============================================================
# WEBSOCKET HANDLER
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 App Android conectada al WebSocket")

    try:
        while True:
            packet_bytes = await websocket.receive_bytes()
            parsed = parse_binary_packet(packet_bytes)
            if not parsed:
                continue

            device_id, timestamps, samples = parsed
            buffers[device_id].extend(zip(timestamps, samples))

            results = align_and_subtract()
            if results:
                insert_processed_data(results)
                await websocket.send_text(f"OK:{len(results)}")
    except Exception as e:
        print("⚠️ Error en WebSocket:", e)
    finally:
        print("❌ App desconectada del WebSocket")

# ============================================================
# CONSULTA REST PARA EL FRONTEND
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
