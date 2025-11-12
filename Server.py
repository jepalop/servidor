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
buffers = {1: [], 2: []}     # cada clave guarda arrays numpy con valores
timestamps_blocks = {1: None, 2: None}  # timestamp inicial del bloque
start_time = None

# ============================================================
# PARSE BINARIO — versión por bloque
# ============================================================
def parse_binary_packet(packet_bytes):
    """
    Decodifica y devuelve los datos de un paquete BLE.
    Devuelve (device_id, ts_start, samples)
    """
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
        samples = np.array(struct.unpack_from(data_fmt, packet_bytes, header_size), dtype=np.float32)

        global start_time
        if start_time is None:
            start_time = datetime.utcnow()
            parse_binary_packet.base_ts = ts_start
            print(f"🕒 Nueva sesión iniciada a {start_time.isoformat()} (base_ts={ts_start})")

        return device_id, ts_start, samples

    except Exception as e:
        print("⚠️ Error parseando paquete:", e)
        return None

# ============================================================
# PROCESAMIENTO POR BLOQUES (sincronía por índice)
# ============================================================
def process_by_sample_index():
    """
    Procesa los bloques de ambos sensores por número de muestra.
    Si ambos tienen un bloque disponible (misma longitud),
    calcula la resta muestra a muestra (front - ref).
    """
    global buffers, timestamps_blocks, start_time

    if len(buffers[1]) == 0 or len(buffers[2]) == 0:
        return None

    # Tomamos el bloque más antiguo de cada uno
    s1 = np.concatenate(buffers[1])
    s2 = np.concatenate(buffers[2])

    # Igualar tamaños (por seguridad, aunque ambos deberían tener 2500)
    n = min(len(s1), len(s2))
    if n == 0:
        return None

    s1 = s1[:n]
    s2 = s2[:n]
    reref = s1 - s2

    # Generar timestamps a partir del bloque más antiguo
    base_ts1 = timestamps_blocks[1]
    base_ts2 = timestamps_blocks[2]
    base_ts = min(base_ts1, base_ts2)

    base_time = start_time + timedelta(milliseconds=(base_ts - getattr(parse_binary_packet, "base_ts", base_ts)))
    timestamps = [base_time + timedelta(milliseconds=i * 1000.0 / FS) for i in range(n)]

    print(f"✅ Bloque procesado: {n} muestras sincronizadas (por índice)")
    buffers = {1: [], 2: []}
    timestamps_blocks = {1: None, 2: None}

    return list(zip(timestamps, reref))

# ============================================================
# GUARDADO EN BASE DE DATOS
# ============================================================
def insert_processed_data(data):
    """Guarda las señales rereferenciadas en la base de datos."""
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
    return {"message": "🧠 Servidor activo — sincronía por número de muestra"}

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

            device_id, ts_start, samples = parsed

            # Guardar bloque completo y su timestamp inicial
            buffers[device_id].append(samples)
            timestamps_blocks[device_id] = ts_start

            # Procesar solo cuando ambos sensores hayan enviado un bloque
            if buffers[1] and buffers[2]:
                results = process_by_sample_index()
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
    """Devuelve los últimos registros rereferenciados (para graficar en frontend)."""
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

