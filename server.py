import base64
import os
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).parent / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).parent / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(__file__).parent / ".cache"))

import cv2
import numpy as np
import json
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

app = FastAPI()
BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=BASE_DIR), name="static")

# Load your custom trained YOLO model here
model = YOLO('yolov8n.pt') # Replace with your 'islet_model.pt'

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # 1. Receive the frame from the browser
            data = await websocket.receive_text()
            
            # 2. Decode the base64 image back into a format OpenCV can read
            encoded_data = data.split(',')[1]
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 3. Run YOLO inference
            results = model(img, verbose=False)
            
            # 4. Extract bounding boxes and count
            detections = json.loads(results[0].to_json())
            islet_count = len(detections)

            # 5. Send the data back to the browser
            await websocket.send_json({
                "count": islet_count,
                "boxes": detections
            })
    except Exception as e:
        print(f"Connection closed: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
