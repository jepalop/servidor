from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import os
import psycopg2

app = FastAPI()

# DB_URL = "postgresql://user_1:password@dpg-xxxx.frankfurt-postgres.render.com:5432/prueba_1_hw7r"
DB_URL = os.getenv("DATABASE_URL")

# Conexión a la base de datos
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

@app.get("/")
def home():
    return {"message": "Servidor WebSocket en FastAPI funcionando"}

# 🔹 Endpoint REST para consultar últimos registros
@app.get("/signals")
def get_signals(limit: int = 20):
    """
    Devuelve los últimos 'limit' registros de la tabla brain_signals
    """
    cursor.execute(
        "SELECT id, timestamp, device_id, data_hex FROM brain_signals ORDER BY id DESC LIMIT %s",
        (limit,)
    )
    rows = cursor.fetchall()
    results = [
        {"id": r[0], "timestamp": r[1].isoformat(), "device_id": r[2], "data_hex": r[3]}
        for r in rows
    ]
    return JSONResponse(content=results)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Cliente conectado")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"Paquete recibido: {data}")

            # Insertar en la DB directamente
            cursor.execute(
                "INSERT INTO brain_signals (device_id, data_hex) VALUES (%s, %s)",
                ("pcb_001", data)
            )
            conn.commit()

            # Respuesta al cliente
            await websocket.send_text("Paquete guardado en DB")
    except Exception as e:
        print(f"Cliente desconectado: {e}")
