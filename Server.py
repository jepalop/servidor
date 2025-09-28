from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import psycopg2
import numpy as np
from datetime import datetime

app = FastAPI()

# 🔹 Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",            # Frontend local
        "https://neuronatech.vercel.app",   # ⚠️ cambia por tu URL en Vercel
        "*",                                # pruebas
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB_URL desde variables de entorno en Render
DB_URL = os.getenv("DATABASE_URL")

conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

@app.get("/")
def home():
    return {"message": "Servidor FastAPI funcionando"}

# 🔹 Endpoint REST para consultar últimos registros
@app.get("/signals")
def get_signals(limit: int = 20):
    cursor.execute(
        """
        SELECT id, timestamp, device_id, value_uv
        FROM brain_signals
        ORDER BY id DESC
        LIMIT %s
        """,
        (limit,)
    )
    rows = cursor.fetchall()
    results = [
        {
            "id": r[0],
            "timestamp": r[1].isoformat() if r[1] else None,
            "device_id": r[2],
            "value_uv": r[3],
        }
        for r in rows
    ]
    return JSONResponse(content=results)

# 🔹 WebSocket para recibir datos binarios
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado")

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                raw_bytes = message["bytes"]
                print(f"Paquete binario recibido: {len(raw_bytes)} bytes")

                # Convertir a array de float32 (µV)
                samples_uv = np.frombuffer(raw_bytes, dtype=np.float32)

                print(f"   → Decodificadas {len(samples_uv)} muestras en µV")

                # Guardar en DB
                for value in samples_uv:
                    cursor.execute(
                        """
                        INSERT INTO brain_signals (device_id, value_uv)
                        VALUES (%s, %s)
                        """,
                        ("pcb_001", float(value))
                    )
                conn.commit()

                await websocket.send_text(f"Guardadas {len(samples_uv)} muestras en DB")

            elif "text" in message and message["text"] is not None:
                data = message["text"]
                cursor.execute(
                    "INSERT INTO brain_signals (device_id, value_uv) VALUES (%s, %s)",
                    ("pcb_001", float(data)),
                )
                conn.commit()
                await websocket.send_text("Texto guardado en DB")

    except Exception as e:
        print(f"Cliente desconectado: {e}")
