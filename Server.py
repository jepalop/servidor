import os
import psycopg2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from scipy.signal import butter, filtfilt, iirnotch
import asyncio

app = FastAPI()

# =====================
# Configuración CORS
# =====================
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

# =====================
# Conexión a PostgreSQL
# =====================
DB_URL = os.getenv("DATABASE_URL")

def connect_db():
    """Abre una nueva conexión y cursor a la base de datos"""
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn, conn.cursor()

conn, cursor = connect_db()

def get_cursor():
    """Verifica que la conexión siga viva; si no, la reabre"""
    global conn, cursor
    try:
        cursor.execute("SELECT 1;")
    except (Exception, psycopg2.Error):
        print("🔄 Reabriendo conexión a PostgreSQL...")
        conn, cursor = connect_db()
    return cursor

# =====================
# Parámetros señal
# =====================
FS = 250  # Hz

def bandpass_filter(data, low=1, high=40, fs=FS, order=4):
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
# Rutas API REST
# =====================
@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    return {"message": "Servidor funcionando ✅"}

@app.get("/signals")
async def get_signals(limit: int = Query(750, ge=1, le=10000)):
    c = get_cursor()
    c.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = c.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": float(r[3])}
        for r in rows
    ]

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

# =====================
# WebSockets
# =====================
clients = set()  # 🔹 clientes frontend suscritos

@app.websocket("/ws")  # PCB -> servidor
async def websocket_pcb(websocket: WebSocket):
    await websocket.accept()
    print("📡 PCB conectado al WebSocket")

    try:
        while True:
            data_bytes = await websocket.receive_bytes()
            print(f"📩 {datetime.now()} - Paquete recibido: {len(data_bytes)} bytes")

            try:
                values = np.frombuffer(data_bytes, dtype="<f4").astype(float).tolist()
            except Exception as e:
                print("❌ Error al decodificar paquete:", e)
                continue

            try:
                c = get_cursor()

                # Batch insert crudos
                c.executemany(
                    "INSERT INTO brain_signals (device_id, value_uv) VALUES (%s, %s)",
                    [("pcb_001", v) for v in values],
                )

                # Filtrar y guardar
                filtered = apply_filters(values)
                c.executemany(
                    "INSERT INTO brain_signals_processed (device_id, value_uv) VALUES (%s, %s)",
                    [("pcb_001", fv) for fv in filtered],
                )

                conn.commit()
                print(f"✅ {datetime.now()} - Guardados {len(values)} crudos y {len(filtered)} filtrados")

            except Exception as e:
                conn.rollback()
                cursor = conn.cursor()
                print("⚠️ Error al insertar en DB:", e)

            # Confirmar a la PCB
            await websocket.send_text(
                f"Guardados {len(values)} crudos y {len(values)} filtrados"
            )

            # 🔹 Reenviar a todos los clientes conectados (filtrados)
            filtered_bytes = np.array(filtered, dtype="<f4").tobytes()
            dead_clients = []
            for client in clients:
                try:
                    await client.send_bytes(filtered_bytes)
                except Exception:
                    dead_clients.append(client)

            for dc in dead_clients:
                clients.remove(dc)

    except Exception as e:
        print("⚠️ Error en WebSocket PCB:", e)
    finally:
        print("❌ PCB desconectado")


@app.websocket("/ws/client")  # frontend -> servidor
async def websocket_client(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("👀 Cliente conectado al WebSocket")

    try:
        while True:
            # 🔹 Enviar keep-alive cada 15 s
            await websocket.send_text("ping")
            await asyncio.sleep(15)
    except Exception as e:
        print("⚠️ Error en WebSocket cliente:", e)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        print("👋 Cliente desconectado")
