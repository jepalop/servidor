from fastapi import FastAPI, WebSocket
import os
import psycopg2

app = FastAPI()

# ⚠️ Pega aquí tu cadena completa de conexión de Render
# DB_URL = "postgresql://user_1:EcA3Rbtd1gZpWUIDWrhIs6kQk168VGx2@dpg-d37dti0gjchc73c73dqg-a.frankfurt-postgres.render.com/prueba_1_hw7r"
DB_URL = os.getenv("DATABASE_URL")

# Conexión a la base de datos
conn = psycopg2.connect(DB_URL)
cursor = conn.cursor()

@app.get("/")
def home():
    return {"message": "Servidor WebSocket en FastAPI funcionando 🚀"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔗 Cliente conectado")

    try:
        while True:
            data = await websocket.receive_text()
            print(f"📩 Paquete recibido: {data}")

            # Insertar en la DB directamente
            cursor.execute(
                "INSERT INTO brain_signals (device_id, data_hex) VALUES (%s, %s)",
                ("pcb_001", data)
            )
            conn.commit()

            # Respuesta al cliente
            await websocket.send_text("✅ Paquete guardado en DB")
    except Exception as e:
        print(f"❌ Cliente desconectado: {e}")
