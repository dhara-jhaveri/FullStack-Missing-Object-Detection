import cv2
import numpy as np
from ultralytics import YOLO
import av
import av.error
import time
import concurrent.futures
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json
import os

# Load YOLOv8 model
model = YOLO('yolov8m.pt')  # Use yolov8n.pt for faster inference if needed

CONF_THRESHOLD = 0.3
RTSP_URL = 'enter your rtsp'
FPS_LIMIT = 10
SKIP_FRAMES = 3
ALERT_COOLDOWN = 1.0
ROI_FILE = 'roi.txt'

def save_roi_to_file(roi_points, filename=ROI_FILE):
    try:
        with open(filename, 'w') as f:
            f.write(f"{roi_points[0][0]},{roi_points[0][1]},{roi_points[1][0]},{roi_points[1][1]}\n")
    except Exception as e:
        print(f"[ERROR] Could not save ROI: {e}")

def load_roi_from_file(filename=ROI_FILE):
    if not os.path.exists(filename):
        return [(1, 1), (4, 4)]  # Default
    try:
        with open(filename, 'r') as f:
            line = f.readline().strip()
            x1, y1, x2, y2 = map(int, line.split(','))
            return [(x1, y1), (x2, y2)]
    except Exception as e:
        print(f"[ERROR] Could not load ROI: {e}")
        return [(1, 1), (4, 4)]  # Default

# Load ROI at startup
roi_points = load_roi_from_file()
roi_lock = asyncio.Lock()
last_alert_time = 0
tracked_box = None  # Box of the initially detected object to track

# For logging last detected object name
default_object_name = None
last_detected_object = None

def log_event(event_type, data):
    with open("events_log.txt", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {event_type} | {data}\n")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if boxAArea + boxBArea - interArea == 0:
        return 0
    return interArea / float(boxAArea + boxBArea - interArea)

def box_inside_roi(box, roi, threshold=0.8):
    xA, yA, xB, yB = box
    rx1, ry1, rx2, ry2 = roi
    inter_x1 = max(xA, rx1)
    inter_y1 = max(yA, ry1)
    inter_x2 = min(xB, rx2)
    inter_y2 = min(yB, ry2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    box_area = (xB - xA) * (yB - yA)
    if box_area == 0:
        return False
    return (inter_area / box_area) >= threshold

def process_frame(frame, x1, y1, x2, y2):
    global last_alert_time, tracked_box, last_detected_object
    roi = (x1, y1, x2, y2)
    results = model(frame, conf=CONF_THRESHOLD)
    boxes = results[0].boxes
    object_present = False
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    if tracked_box is None:
        # Detect and lock onto first object inside ROI
        for box in boxes:
            xA, yA, xB, yB = map(int, box.xyxy[0])
            if box_inside_roi([xA, yA, xB, yB], roi):
                tracked_box = [xA, yA, xB, yB]
                label = f"Tracking: {model.names[int(box.cls[0])]}"
                last_detected_object = model.names[int(box.cls[0])]
                cv2.rectangle(frame, (xA, yA), (xB, yB), (255, 255, 0), 2)
                cv2.putText(frame, label, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                object_present = True
                break
    else:
        # Check if the same object is still visible
        for box in boxes:
            xA, yA, xB, yB = map(int, box.xyxy[0])
            iou = compute_iou(tracked_box, [xA, yA, xB, yB])
            if iou > 0.5:
                object_present = True
                label = f"Object Present"
                last_detected_object = model.names[int(box.cls[0])]
                cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
                cv2.putText(frame, label, (xA, yA - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                tracked_box = [xA, yA, xB, yB]
                break

    if not object_present:
        if time.time() - last_alert_time > ALERT_COOLDOWN:
            print("[ALERT] Object moved/missing from ROI!")
            last_alert_time = time.time()
            log_event(
                "OBJECT_MISSING",
                f"ROI: ({x1},{y1}) to ({x2},{y2}), Last detected object: {last_detected_object}"
            )
        cv2.putText(frame, "OBJECT MISSING!", (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, object_present

async def frame_generator():
    container = None
    try:
        container = av.open(RTSP_URL, options={'rtsp_transport': 'tcp', 'stimeout': '10000000'})
        video_stream = next(s for s in container.streams if s.type == 'video')
        last_processed_time = 0
        frame_counter = 0
        fps = 0
        last_time = time.time()
        while True:
            async with roi_lock:
                x1, y1 = roi_points[0]
                x2, y2 = roi_points[1]
            for packet in container.demux(video_stream):
                for frame_pyav in packet.decode():
                    current_time = time.time()
                    if current_time - last_processed_time < 1.0 / FPS_LIMIT:
                        await asyncio.sleep(0.001)
                        continue
                    last_processed_time = current_time
                    frame = frame_pyav.to_ndarray(format='bgr24')
                    frame = cv2.resize(frame, (960, 600))
                    frame_counter += 1
                    if frame_counter % SKIP_FRAMES != 0:
                        continue
                    now = time.time()
                    fps = 0.9 * fps + 0.1 * (1 / (now - last_time)) if frame_counter > 1 else 0
                    last_time = now
                    processed_frame, object_present = process_frame(frame.copy(), x1, y1, x2, y2)
                    ret, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                    if not ret:
                        continue
                    yield buffer.tobytes(), object_present, fps
    except av.error.FFmpegError as e:
        print(f"[ERROR] FFmpeg Error: {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
    finally:
        if container:
            container.close()

@app.websocket("/ws/stream")
async def video_stream(websocket: WebSocket):
    global tracked_box
    await websocket.accept()
    frame_task = asyncio.create_task(frame_sender(websocket))
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get('type') == 'roi' and 'roi' in msg:
                    roi = msg['roi']
                    async with roi_lock:
                        roi_points[0] = (int(roi['x1']), int(roi['y1']))
                        roi_points[1] = (int(roi['x2']), int(roi['y2']))
                        save_roi_to_file(roi_points)  # Save ROI when updated
                    log_event("ROI_SELECTED", f"Coordinates: {roi_points[0]} to {roi_points[1]}")
                    tracked_box = None  # Reset tracking when ROI is updated
                elif msg.get('type') == 'clear_roi':
                    async with roi_lock:
                        roi_points[0] = (1, 1)
                        roi_points[1] = (4, 4)
                        save_roi_to_file(roi_points)  # Save default ROI
                    log_event("ROI_CLEARED", "ROI reset to default (1,1) to (4,4)")
                    tracked_box = None  # Reset tracking when ROI is cleared
            except Exception as e:
                print(f"Error parsing ROI message: {e}")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        frame_task.cancel()
        await websocket.close()

async def frame_sender(websocket: WebSocket):
    async for frame_bytes, object_present, fps in frame_generator():
        await websocket.send_json({
            "type": "frame",
            "data": frame_bytes.hex(),
            "stats": {
                "object_present": object_present,
                "fps": fps
            }
        })

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
