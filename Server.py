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
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

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
@app.get("/")
async def root():
    return {"message": "Servidor funcionando ✅"}

@app.get("/signals")
async def get_signals(limit: int = Query(750, ge=1, le=10000)):
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": float(r[3])}
        for r in rows
    ]

@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(750, ge=1, le=10000)):
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals_processed ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
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
            # Recibir paquete binario desde la PCB
            data_bytes = await websocket.receive_bytes()
            print(f"📩 {datetime.now()} - Paquete recibido: {len(data_bytes)} bytes")

            try:
                # Decodificar como float32 little-endian
                values = np.frombuffer(data_bytes, dtype="<f4").astype(float).tolist()
            except Exception as e:
                print("❌ Error al decodificar paquete:", e)
                continue

            try:
                # Guardar señal cruda
                for v in values:
                    cursor.execute(
                        "INSERT INTO brain_signals (device_id, value_uv) VALUES (%s, %s)",
                        ("pcb_001", v),
                    )

                # Filtrar y guardar
                filtered = apply_filters(values)
                for fv in filtered:
                    cursor.execute(
                        "INSERT INTO brain_signals_processed (device_id, value_uv) VALUES (%s, %s)",
                        ("pcb_001", fv),
                    )

                conn.commit()
                print(f"✅ {datetime.now()} - Guardados {len(values)} crudos y {len(filtered)} filtrados")
            except Exception as e:
                conn.rollback()
                print("⚠️ Error al insertar en DB:", e)

            # Confirmar a la PCB
            await websocket.send_text(
                f"Guardados {len(values)} crudos y {len(values)} filtrados"
            )

            # 🔹 Reenviar a todos los clientes conectados al frontend
            dead_clients = []
            for client in clients:
                try:
                    await client.send_bytes(data_bytes)
                except:
                    dead_clients.append(client)

            # limpiar clientes desconectados
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
            # Mantener la conexión abierta
            await asyncio.sleep(1)
    except Exception as e:
        print("⚠️ Error en WebSocket cliente:", e)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        print("👋 Cliente desconectado")
