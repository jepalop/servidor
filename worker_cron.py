import psycopg2
import os
import numpy as np
from filters import process_ch1

DB_URL = os.getenv("DATABASE_URL")

def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Leer 2500 últimas muestras de CH1
    cur.execute("""
        SELECT id, value_uv, device_id, timestamp
        FROM brain_signals
        ORDER BY id DESC LIMIT 2500
    """)
    rows = cur.fetchall()

    if not rows:
        print("⚠️ No hay datos nuevos para procesar")
        return

    # Extraer CH1 en µV
    ch1 = np.array([r[1] for r in rows], dtype=float)

    # Procesar
    results = process_ch1(ch1)

    # Guardar resultados en tabla brain_signals_processed
    cur.execute("""
        INSERT INTO brain_signals_processed (timestamp, device_id, filtered, fft)
        VALUES (%s, %s, %s, %s)
    """, (
        rows[-1][3],   # timestamp del último registro
        rows[-1][2],   # device_id
        results["ch1_filtered"],
        results["fft"]
    ))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Procesados {len(rows)} registros")

if __name__ == "__main__":
    main()
