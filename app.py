import asyncio
import websockets
import json
import cv2
import numpy as np
import base64
import asyncpg
import logging
from datetime import datetime  # <-- TAMBAHAN BARU
from ultralytics import YOLO

# ─── KONFIGURASI LOGGING ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='\n%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("BackendYOLO")

# ─── LOAD MODEL ───────────────────────────────────────────────────────────────
logger.info("Loading YOLO model 'best_openvino_model'...")
model = YOLO('best_openvino_model', task='detect')
logger.info("YOLO model loaded successfully.")

# ─── KONFIGURASI DATABASE ───────────────────────────────────────────────────
DB_CONFIG = {
    "user": "postgres",
    "password": "k}=>.(=+8,Nq3KkR", 
    "database": "mbasystem",                  
    "host": "34.59.60.237", 
    "port": 5432                              
}

async def process_frame(websocket, db_pool):
    client_ip = websocket.remote_address[0]
    logger.info(f"Frontend client connected from {client_ip}")
    
    try:
        async for message in websocket:
            # 1. CEK JENIS PESAN (JSON untuk Data Tracking dari Frontend)
            if message.startswith("{"):
                try:
                    payload = json.loads(message)
                    if payload.get("type") == "SAVE_TRACKING":
                        data_length = len(payload.get("data", []))
                        logger.info(f"Received request to save {data_length} transition segment(s).")
                        
                        # Insert secara asinkron
                        await save_to_db(db_pool, payload["data"])
                except json.JSONDecodeError:
                    logger.error("Failed to decode JSON payload from frontend.")
                continue

            # 2. FRAME GAMBAR BASE64 UNTUK YOLO
            try:
                encoded_data = message.split(',')[1]
                nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                results = model.predict(source=frame, conf=0.5, verbose=False)
                
                detections = []
                if len(results) > 0:
                    boxes = results[0].boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        class_name = model.names[int(box.cls[0])]
                        
                        detections.append({
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "conf": conf, "class_name": class_name
                        })

                await websocket.send(json.dumps({"detections": detections}))
            except Exception as e:
                logger.error(f"Error processing video frame: {e}")
                
    except websockets.exceptions.ConnectionClosed as e:
        logger.warning(f"Frontend client {client_ip} disconnected. Reason: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in WebSocket connection: {e}")

async def save_to_db(pool, tracking_data_list):
    if not tracking_data_list:
        return

    query = """
        INSERT INTO multimodal_tracking 
        (camera_id, tracking_id, emotion, is_attentive, yaw, pitch, yolo_action, action_conf, start_time, end_time, duration)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
    """
    
    values = []
    for d in tracking_data_list:
        # 🚨 PERBAIKAN WAKTU: Konversi UTC (Aware) ke Waktu Lokal (Naive)
        start_dt = datetime.fromisoformat(d["start_time"].replace('Z', '+00:00')).astimezone().replace(tzinfo=None)
        end_dt = datetime.fromisoformat(d["end_time"].replace('Z', '+00:00')).astimezone().replace(tzinfo=None)
        
        values.append((
            d["camera_id"], d["tracking_id"], d["emotion"], d["is_attentive"],
            d["yaw"], d["pitch"], d["yolo_action"], d["action_conf"],
            start_dt, end_dt, d["duration"]
        ))

    try:
        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, values)
                logger.info(f"Successfully inserted {len(values)} segment(s) into PostgreSQL.")
    except asyncpg.exceptions.PostgresError as e:
        logger.error(f"Database insertion failed (PostgresError): {e}")
    except Exception as e:
        logger.error(f"Database insertion failed (Unexpected Error): {e}")

async def main():
    logger.info("Initializing Database Connection Pool...")
    try:
        db_pool = await asyncpg.create_pool(**DB_CONFIG)
        logger.info("Database Connection Pool created successfully.")
    except Exception as e:
        logger.critical(f"Failed to connect to PostgreSQL: {e}")
        return

    bound_handler = lambda ws: process_frame(ws, db_pool)
    
    try:
        # 🚨 PERBAIKAN: Tambahkan max_size dan ping_interval agar koneksi stabil
        # max_size=None berarti tidak ada batasan ukuran file gambar yang dikirim
        async with websockets.serve(bound_handler, "localhost", 8765, max_size=None, ping_interval=None):
            logger.info("==================================================")
            logger.info(" YOLO & DB WebSocket Server is RUNNING ")
            logger.info(" Listening on: ws://localhost:8765 ")
            logger.info("==================================================")
            await asyncio.Future()  # Run forever
    except Exception as e:
        logger.critical(f"Failed to start WebSocket server: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server manually stopped by user (KeyboardInterrupt).")