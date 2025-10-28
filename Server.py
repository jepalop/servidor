import os
import psycopg2
import numpy as np
import json
from datetime import datetime
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, iirnotch
import asyncio

app = FastAPI()

# ============================================================
# CORS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Conexión a PostgreSQL
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
        print("🔄 Reabriendo conexión a PostgreSQL...")
        conn, cursor = connect_db()
    return cursor

# ============================================================
# Parámetros de señal
# ============================================================
FS = 250  # Hz

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
# API REST
# ============================================================
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Servidor funcionando ✅"}

@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(750, ge=1, le=10000)):
    c = get_cursor()
    c.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals_processed ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = c.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": float(r[3])}
        for r in rows
    ]

# ============================================================
# WebSockets
# ============================================================
clients = set()

@app.websocket("/ws")
async def websocket_pcb(websocket: WebSocket):
    await websocket.accept()
    print("📡 App Android conectada al WebSocket")

    try:
        while True:
            message = await websocket.receive_text()
            timestamp_received = datetime.now()

            try:
                data = json.loads(message)
                timestamp_start = data.get("timestamp_start")
                channels = data.get("channels", {})
                front = channels.get("Neurona_front", [])
                ref = channels.get("Neurona_ref", [])

                # Verificación
                if not front or not ref:
                    print("⚠️ Paquete JSON incompleto (faltan canales)")
                    continue

                # Igualar longitudes
                n = min(len(front), len(ref))
                front, ref = front[:n], ref[:n]

                # 🔹 Convertir a float antes de operaciones NumPy
                front = np.array([float(x) for x in front], dtype=np.float32)
                ref = np.array([float(x) for x in ref], dtype=np.float32)

                # Re-referenciar
                reref = np.subtract(front, ref)

                # Filtrar señal
                filtered_reref = apply_filters(reref)

                # Calcular timestamps individuales
                timestamps = [
                    datetime.fromtimestamp((timestamp_start / 1000.0) + i / FS)
                    for i in range(len(filtered_reref))
                ]

                # Guardar en base de datos
                c = get_cursor()
                c.executemany(
                    "INSERT INTO brain_signals_processed (device_id, timestamp, value_uv) VALUES (%s, %s, %s)",
                    [("reref", ts, float(v)) for ts, v in zip(timestamps, filtered_reref)],
                )
                conn.commit()

                print(f"✅ {timestamp_received} - Guardadas {len(filtered_reref)} muestras re-referenciadas")

                # Confirmar a la app
                await websocket.send_text(f"Guardadas {len(filtered_reref)} muestras procesadas")

                # Enviar a clientes conectados (si hay dashboard, por ejemplo)
                filtered_bytes = np.array(filtered_reref, dtype="<f4").tobytes()
                dead_clients = []
                for client in clients:
                    try:
                        await client.send_bytes(filtered_bytes)
                    except:
                        dead_clients.append(client)
                for dc in dead_clients:
                    clients.remove(dc)

            except json.JSONDecodeError:
                print("⚠️ Error: paquete no es JSON válido")
            except Exception as e:
                conn.rollback()
                print("⚠️ Error procesando paquete:", e)

    except Exception as e:
        print(f"⚠️ Error en WebSocket principal: {e}")
    finally:
        print("❌ App desconectada del WebSocket")

@app.websocket("/ws/client")
async def websocket_client(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("👀 Cliente conectado al WebSocket")

    try:
        while True:
            await websocket.send_text("ping")
            await asyncio.sleep(15)
    except Exception as e:
        print("⚠️ Error en WebSocket cliente:", e)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        print("👋 Cliente desconectado")
