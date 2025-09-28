from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import psycopg2

app = FastAPI()

# 🔹 Configuración de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",          # Frontend local
        "https://neuronatech.vercel.app", # ⚠️ cambia por la URL real en Vercel
        "*",  # Para pruebas, puedes quitarlo en producción
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DB_URL desde variables de entorno en Render
DB_URL = os.getenv("DATABASE_URL")

# Conexión a la base de datos
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

@app.get("/")
def home():
    return {"message": "Servidor FastAPI funcionando ✅"}

# 🔹 Endpoint REST para consultar últimos registros
@app.get("/signals")
def get_signals(limit: int = 20):
    """
    Devuelve los últimos 'limit' registros de la tabla brain_signals,
    mostrando directamente el valor en µV.
    """
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
        {"id": r[0], "timestamp": r[1].isoformat(), "device_id": r[2], "value_uv": r[3]}
        for r in rows
    ]
    return JSONResponse(content=results)

# 🔹 WebSocket (para datos entrantes de la app Android)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"Paquete recibido: {data}")

            # Insertar en la DB → ahora ya no se guarda el HEX, sino que
            # debería ser un valor ya convertido (µV) o JSON con lotes.
            cursor.execute(
                "INSERT INTO brain_signals (device_id, data_hex) VALUES (%s, %s)",
                ("pcb_001", data)
            )
            conn.commit()

            await websocket.send_text("Paquete guardado en DB")
    except Exception as e:
        print(f"Cliente desconectado: {e}")
