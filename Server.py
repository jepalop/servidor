import os
import psycopg2
import numpy as np
import asyncio
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from filters import bandpass_filter, notch_filter  # 👈 tus funciones de filtrado

app = FastAPI()

# =====================
# Configuración CORS
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://neuronatech.vercel.app", "*"],
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
# Rutas API
# =====================
@app.get("/")
async def root():
    return {"message": "Servidor funcionando ✅"}

@app.get("/signals")
async def get_signals(limit: int = Query(2500, ge=1, le=10000)):
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": r[3]}
        for r in rows
    ]

@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(2500, ge=1, le=10000)):
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals_processed ORDER BY id DESC LIMIT %s;",
        (limit,),
    )
    rows = cursor.fetchall()
    return [
        {"id": r[0], "timestamp": r[1], "device_id": r[2], "value_uv": r[3]}
        for r in rows
    ]

# =====================
# WebSocket
# =====================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📡 Cliente conectado al WebSocket")

    try:
        while True:
            data_bytes = await websocket.receive_bytes()
            print(f"📩 Paquete binario recibido: {len(data_bytes)} bytes")

            try:
                values = np.frombuffer(data_bytes, dtype=np.float32).tolist()
            except Exception as e:
                print("❌ Error al decodificar paquete:", e)
                continue

            for v in values:
                cursor.execute(
                    "INSERT INTO brain_signals (device_id, value_uv) VALUES (%s, %s)",
                    ("pcb_001", v),
                )
            conn.commit()

            print(f"✅ Guardados {len(values)} valores en la DB")
            await websocket.send_text(f"Guardados {len(values)} valores en la DB")
    except Exception as e:
        print("⚠️ Error en WebSocket:", e)
    finally:
        await websocket.close()
        print("❌ Cliente desconectado")

# =====================
# Tarea de filtrado en segundo plano
# =====================
async def process_loop():
    while True:
        try:
            cursor.execute(
                "SELECT value_uv FROM brain_signals ORDER BY id DESC LIMIT 2500;"
            )
            rows = cursor.fetchall()
            if rows:
                raw = np.array([r[0] for r in rows], dtype=np.float32)

                # aplicar filtros
                filtered = bandpass_filter(raw, fs=250, low=1, high=40)
                filtered = notch_filter(filtered, fs=250, freq=50)

                # guardar en DB
                for v in filtered:
                    cursor.execute(
                        "INSERT INTO brain_signals_processed (device_id, value_uv) VALUES (%s, %s)",
                        ("pcb_001", float(v)),
                    )
                conn.commit()
                print(f"✅ Filtrados {len(filtered)} valores y guardados en brain_signals_processed")
        except Exception as e:
            print("❌ Error en loop de filtrado:", e)

        await asyncio.sleep(10)  # repetir cada 10s

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(process_loop())
