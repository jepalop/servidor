import os
import psycopg2
import numpy as np
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket, Query
from fastapi.middleware.cors import CORSMiddleware
from psycopg2.extras import execute_batch

# ============================================================
# FASTAPI SETUP
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATABASE
# ============================================================
DB_URL = os.getenv("DATABASE_URL")

def connect_db():
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn, conn.cursor()

conn, cursor = connect_db()

def get_cursor():
    """Verifica que la conexión con PostgreSQL siga activa."""
    global conn, cursor
    try:
        cursor.execute("SELECT 1;")
    except (Exception, psycopg2.Error):
        print("🔄 Reconnecting to PostgreSQL...")
        conn, cursor = connect_db()
    return cursor

# ============================================================
# VARIABLES GLOBALES DE SINCRONIZACIÓN
# ============================================================
FS = 250  # Frecuencia de muestreo (Hz)
start_time = None
sample_counter = 0

# ============================================================
# ENDPOINT RAÍZ
# ============================================================
@app.get("/")
async def root():
    return {"message": "🧠 Servidor activo y listo (solo re-referenciado, sin filtrado)"}

# ============================================================
# WEBSOCKET — RECEPCIÓN DE DATOS BINARIOS
# ============================================================
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global start_time, sample_counter
    await websocket.accept()
    print("📡 App Android conectada al WebSocket")

    try:
        while True:
            data = await websocket.receive_bytes()

            try:
                floats = np.frombuffer(data, dtype="<f4")
                if len(floats) % 2 != 0:
                    print("⚠️ Paquete con longitud inválida, ignorado")
                    continue

                pairs = floats.reshape(-1, 2)
                front = pairs[:, 0]
                ref = pairs[:, 1]

                # ============================================================
                # 🔧 Alineación automática por correlación cruzada
                # ============================================================
                corr = np.correlate(front - np.mean(front), ref - np.mean(ref), mode="full")
                lag = np.argmax(corr) - (len(ref) - 1)

                if abs(lag) > 0:
                    if lag > 0:
                        front = front[lag:]
                        ref = ref[:len(front)]
                    elif lag < 0:
                        ref = ref[-lag:]
                        front = front[:len(ref)]
                    print(f"🔧 Ajuste de desfase: {lag} muestras ({lag / FS:.3f}s)")

                # ============================================================
                # 🧮 Re-referenciado (Front - Ref)
                # ============================================================
                reref = front - ref

                # ============================================================
                # 🕒 Timestamps continuos
                # ============================================================
                if start_time is None:
                    start_time = datetime.now()
                    sample_counter = 0

                timestamps = [
                    start_time + timedelta(seconds=(sample_counter + i) / FS)
                    for i in range(len(reref))
                ]
                sample_counter += len(reref)

                # ============================================================
                # 💾 Inserción a base de datos
                # ============================================================
                c = get_cursor()
                execute_batch(
                    c,
                    """
                    INSERT INTO brain_signals_processed (device_id, timestamp, value_uv)
                    VALUES (%s, %s, %s)
                    """,
                    [("reref", ts, float(v)) for ts, v in zip(timestamps, reref)],
                    page_size=500,
                )
                conn.commit()

                print(
                    f"✅ {len(reref)} muestras procesadas "
                    f"(lag={lag}, t0={timestamps[0].strftime('%H:%M:%S.%f')[:-3]})"
                )

                await websocket.send_text(f"OK:{len(reref)}")

            except Exception as e:
                conn.rollback()
                print("⚠️ Error procesando paquete:", e)
                await websocket.send_text(f"ERROR:{e}")

    except Exception as e:
        print("⚠️ Error en WebSocket principal:", e)

    finally:
        print("❌ App desconectada del WebSocket")

# ============================================================
# ENDPOINT REST — CONSULTA PARA EL FRONTEND
# ============================================================
@app.get("/signals/processed")
async def get_signals_processed(limit: int = Query(750, ge=1, le=10000)):
    """Devuelve las últimas muestras procesadas para el frontend."""
    c = get_cursor()
    c.execute(
        """
        SELECT id, timestamp, device_id, value_uv
        FROM brain_signals_processed
        ORDER BY id DESC
        LIMIT %s;
        """,
        (limit,),
    )
    rows = c.fetchall()
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "device_id": r[2],
            "value_uv": float(r[3]),
        }
        for r in rows
    ]
