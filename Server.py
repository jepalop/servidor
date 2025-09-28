import os
import psycopg2
import numpy as np
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# =====================
# Configuración CORS
# =====================
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",          # Frontend local
        "https://neuronatech.vercel.app", # URL en Vercel
        "*",  # Para pruebas, quitar en producción
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
# Rutas API
# =====================
@app.get("/")
async def root():
    return {"message": "Servidor funcionando ✅"}

@app.get("/signals")
async def get_signals(limit: int = Query(2500, ge=1, le=10000)):
    """
    Devuelve los últimos 'limit' registros.
    Por defecto: 2500 (≈10s a 250 Hz).
    """
    cursor.execute(
        "SELECT id, timestamp, device_id, value_uv FROM brain_signals ORDER BY id DESC LIMIT %s;",
        (limit,)
    )
    rows = cursor.fetchall()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "device_id": r[2],
            "value_uv": r[3],
        }
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
            data_bytes = await websocket.receive_bytes()  # recibir binario
            print(f"📩 Paquete binario recibido: {len(data_bytes)} bytes")

            try:
                values = np.frombuffer(data_bytes, dtype=np.float32).tolist()
            except Exception as e:
                print("❌ Error al decodificar paquete:", e)
                continue

            # Guardar cada valor en la base de datos
            for v in values:
                cursor.execute(
                    "INSERT INTO brain_signals (device_id, value_uv) VALUES (%s, %s)",
                    ("pcb_001", v)
                )
            conn.commit()

            print(f"✅ Guardados {len(values)} valores en la DB")

            await websocket.send_text(f"Guardados {len(values)} valores en la DB")
    except Exception as e:
        print("⚠️ Error en WebSocket:", e)
    finally:
        await websocket.close()
        print("❌ Cliente desconectado")
