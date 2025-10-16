import os
import psycopg2
import numpy as np
from datetime import datetime
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
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
# PARÁMETROS BÁSICOS
# ============================================================
FS = 250  # Hz (para referencia futura)

# ============================================================
# ENDPOINTS DE API
# ============================================================
@app.get("/")
async def root():
    return {"message": "Servidor funcionando"}


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
    print("PCB conectada al WebSocket")

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
                ref = np.median(np.vstack([ch1, ch2]), axis=0)
                y_raw = ch1 - ref

                cursor.executemany(
                    "INSERT INTO brain_signals_processed (device_id, value_uv) VALUES (%s, %s)",
                    [("pcb_001", float(v)) for v in y_raw],
                )
                conn.commit()

                print(f"✅ {datetime.now()} - Guardadas {len(y_raw)} muestras re-referenciadas robustamente")

            except Exception as e:
                conn.rollback()
                print("⚠️ Error durante guardado en base de datos:", e)
                continue

            # Confirmación a la app Android
            await websocket.send_text(
                f"Guardadas {len(y_raw)} muestras re-referenciadas"
            )

            dead_clients = []
            for client in clients:
                try:
                    await client.send_bytes(y_raw.astype("<f4").tobytes())
                except:
                    dead_clients.append(client)

            for dc in dead_clients:
                clients.remove(dc)

    except Exception as e:
        print("Error WebSocket PCB:", e)
    finally:
        print("PCB desconectada")


@app.websocket("/ws/client")  # frontend -> servidor
async def websocket_client(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    print("Cliente conectado al WebSocket")

    try:
        while True:
            await asyncio.sleep(1)
    except Exception as e:
        print("Error WebSocket cliente:", e)
    finally:
        if websocket in clients:
            clients.remove(websocket)
        print("Cliente desconectado")
