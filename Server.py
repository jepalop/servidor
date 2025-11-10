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
FS = 250  # Hz
buffers = {1: [], 2: []}
start_time = None
sample_counter = 0

# ============================================================
# PARSE BINARIO (corregido + debug)
# ============================================================
def parse_binary_packet(packet_bytes):
    """Decodifica paquete binario proveniente de la app Android."""
    try:
        header_fmt = "<Bqhh"  # [device_id][timestamp_start][sample_rate][n_samples]
        header_size = struct.calcsize(header_fmt)
        if len(packet_bytes) < header_size:
            print("⚠️ Paquete demasiado corto para cabecera")
            return None

        device_id, ts_start, fs, n = struct.unpack_from(header_fmt, packet_bytes, 0)
        expected_size = header_size + 4 * n
        if len(packet_bytes) != expected_size:
            print(f"⚠️ Tamaño inesperado ({len(packet_bytes)} vs {expected_size})")
            return None

        data_fmt = f"<{n}f"
        samples = struct.unpack_from(data_fmt, packet_bytes, header_size)

        # 🧠 Si es la primera vez, fijamos tiempo base (hora actual UTC)
        global start_time
        if start_time is None:
            start_time = datetime.utcnow()
            print(f"🕒 Sesión iniciada: {start_time.isoformat()}")

        # Calcular timestamps relativos a la sesión
        timestamps = [
            start_time + timedelta(seconds=(ts_start / 1000.0 - start_time.timestamp()) + i / fs)
            for i in range(n)
        ]

        print(f"📦 Paquete decodificado — Device {device_id} | {n} muestras | t0={timestamps[0].time()}")

        return device_id, timestamps, samples

    except Exception as e:
        print("⚠️ Error parseando paquete:", e)
        return None

# ============================================================
# PROCESAMIENTO Y GUARDADO
# ============================================================
def align_and_subtract():
    """
    Alinea los buffers de Neurona_front (1) y Neurona_ref (2)
    y calcula la señal rereferenciada.
    """
    global buffers

    if not buffers[1] or not buffers[2]:
        return None

    t1 = buffers[1][0][0]
    t2 = buffers[2][0][0]
    dt = abs((t1 - t2).total_seconds())

    # Debug para ver desalineaciones
    if dt > 0.001:
        print(f"⚠️ Desalineación temporal detectada: Δt={dt*1000:.2f} ms")

    # Espera si aún no están alineadas dentro de 20 ms
    if dt > 0.02:
        return None

    n = min(len(buffers[1]), len(buffers[2]))
    t1, v1 = zip(*buffers[1][:n])
    t2, v2 = zip(*buffers[2][:n])

    reref = np.array(v1) - np.array(v2)
    timestamps = t1[:n]

    # Vaciar los fragmentos usados
    buffers[1] = buffers[1][n:]
    buffers[2] = buffers[2][n:]

    print(f"✅ Alineadas {n} muestras — front:{len(buffers[1])} / ref:{len(buffers[2])} restantes")

    return list(zip(timestamps, reref))

def insert_processed_data(data):
    """Inserta en PostgreSQL las señales rereferenciadas."""
    try:
        c = get_cursor()
        execute_batch(
            c,
            """
            INSERT INTO brain_signals_processed (device_id, timestamp, value_uv)
            VALUES (%s, %s, %s)
            """,
            [("reref", ts, float(v)) for ts, v in data],
            page_size=500,
        )
        conn.commit()
        print(f"💾 Guardadas {len(data)} muestras procesadas")
    except Exception as e:
        conn.rollback()
        print("⚠️ Error al guardar en la BD:", e)

# ============================================================
# ENDPOINT ROOT
# ============================================================
@app.get("/")
async def root():
    return {"message": "🧠 Servidor activo y listo (sincronización continua habilitada)"}

# ============================================================
# WEBSOCKET PRINCIPAL
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 App Android conectada al WebSocket")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"]:
                packet_bytes = message["bytes"]
            elif "text" in message and message["text"]:
                packet_bytes = message["text"].encode("latin-1")
            else:
                continue

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
# ENDPOINT REST PARA FRONTEND
# ============================================================
@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(750, ge=1, le=10000)):
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
