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
buffers = {1: None, 2: None}
start_time = None

# ============================================================
# PARSE BINARIO — formato por bloque
# ============================================================
def parse_binary_packet(packet_bytes):
    """
    Devuelve (device_id, samples) desde un paquete BLE.
    Ignora los timestamps, se sincroniza por índice.
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
        samples = np.frombuffer(packet_bytes, dtype=np.float32, offset=header_size, count=n)
        return device_id, samples

    except Exception as e:
        print("⚠️ Error parseando paquete:", e)
        return None

# ============================================================
# PROCESAMIENTO POR MUESTRA
# ============================================================
def process_by_sample_index():
    """
    Combina los buffers de ambos sensores por índice (0..N-1).
    Genera una señal rereferenciada front - ref.
    """
    global buffers, start_time

    s1 = buffers[1]
    s2 = buffers[2]
    if s1 is None or s2 is None:
        return None

    n = min(len(s1), len(s2))
    if n == 0:
        return None

    reref = s1[:n] - s2[:n]

    # Generar timestamps relativos solo para referencia visual
    if start_time is None:
        start_time = datetime.utcnow()
    timestamps = [start_time + timedelta(milliseconds=i * 1000.0 / FS) for i in range(n)]

    # Reiniciar buffers
    buffers = {1: None, 2: None}
    print(f"✅ Bloque procesado por índice: {n} muestras rereferenciadas")

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
            [(0, ts, float(v)) for ts, v in data],  # device_id=0 → reref
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

            packet_bytes = message.get("bytes") or message.get("text", "").encode("latin-1")
            if not packet_bytes:
                continue

            parsed = parse_binary_packet(packet_bytes)
            if not parsed:
                continue

            device_id, samples = parsed
            buffers[device_id] = samples

            # Procesar cuando ambos sensores hayan enviado su bloque
            if buffers[1] is not None and buffers[2] is not None:
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

