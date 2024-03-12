import cv2
import base64
import io
import torch
from PIL import Image
import numpy as np
from flask_socketio import emit, SocketIO
from flask import Flask, render_template
from ultralytics import YOLO


app = Flask(__name__, template_folder="templates")
sio = SocketIO(app, cors_allowed_origins="*")

@sio.on('connect')
def test_connect():
    emit('my response', {'data': 'Connected'})

# ------------------------ YOLO Model Loading ------------------------
def load_yolo_model(model_path='/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/api/flaskProject/model/best20.pt', conf_thres=0.25, iou_thres=0.45):
    """
    Loads a custom YOLO model from a local path.

    Args:
        model_path (str): Path to the pre-trained YOLO model weights (.pt file).
        conf_thres (float): Object confidence threshold for detections. Default is 0.25.
        iou_thres (float): IoU threshold for non-maximum suppression. Default is 0.45.

    Returns:
        torch.nn.Module: The loaded YOLO model.
    """

    try:
        model = YOLO(model_path)
        model.conf = conf_thres  # Set confidence threshold
        model.iou = iou_thres    # Set IoU threshold
        return model
    except Exception as e:
        print(f"Error loading custom YOLO model: {e}")
        return None  # Return None on failure

# ------------------------ Image Processing Function ------------------------
@sio.on('frame')
def image(data_image):
    if 'data' not in data_image:
        print("Error: 'data' key not found in received data")
        return

    data_string = data_image['data']

    try:
        # Decode base64 data
        img_bytes = base64.b64decode(data_string)

        # Load image using Pillow (PIL)
        pimg = Image.open(io.BytesIO(img_bytes))

        # Convert to OpenCV format (numpy array)
        frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
        frame = cv2.flip(frame, flipCode=0)

        # --- YOLO Integration ---
        model = load_yolo_model()

        # Inference
        results = model(frame)

        # Process detections
        detected_classes = []
        for detection in results.xyxy[0]:
            confidence, x0, y0, x1, y1, class_id = detection.tolist()
            if confidence > 0.5:
                class_name = model.names[int(class_id)]
                x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)  # Convert to integers
                print(f"Detected Class: {class_name}, Bounding Box: ({x0}, {y0}), ({x1}, {y1})")

        # --- Response ---
        if detected_classes:
            print(f"Detected Classes: {detected_classes}")

        # (Optional) Modify if you want to include the image in the response
        emit('response_back', {'classes': detected_classes})

    except Exception as e:
        print(f"Error processing image: {e}")
        emit('error', str(e))

@sio.on('message')
def handle_message(message):
    # logging.info("The message received: %s", message)
    print(f'Received message: {message}')
    emit('response', {'status': 'Text message received'})



if __name__ == "__main__":
    sio.run(app, debug=True)