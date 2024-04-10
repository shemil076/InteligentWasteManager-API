from flask import Flask
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import io
from PIL import Image

from ultralytics import YOLO

from app.models.frame import Frame

app = Flask(__name__)
sio = SocketIO(app, cors_allowed_origins="*")

frames = []

model_path = ('/Users/pramudithakarunarathna/Documents/IIT BEng Software Engineering/FYP/Implementations/flaskApi/app/detectionModel/originalYolo.pt')
print("Model is loading...")
model = YOLO(model_path)


@sio.on('connect')
def test_connect():
    # load_model()
    emit('my response', {'data': 'Connected'})


def decode_image(data_string):
    img_bytes = base64.b64decode(data_string)
    pimg = Image.open(io.BytesIO(img_bytes))
    frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
    return frame


def detect_waste_objects(frame):
    global model
    results = model.predict(frame, conf=0.48)

    detected_classes = []

    for result in results:
        detected_classes.append(result.tojson())

    detected_classes = detected_classes[0].replace('\n', '')
    return detected_classes


@sio.on('frame')
def handle_frame(data_image):
    if 'data' not in data_image:
        emit('error', {'message': "Missing 'data' key in received data"})
        return
    try:
        frame = decode_image(data_image['data'])
        frames.append(Frame(data_image['data']))

        detection_results = detect_waste_objects(frame)

        emit('detection_result', {'detected_objects': detection_results})
    except Exception as e:
        print("exception occurred", e)
        emit('error', {'message': str(e)})


if __name__ == "__main__":
    sio.run(app, debug=True, host='0.0.0.0', port=5000)
