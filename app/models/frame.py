import base64

import cv2
import io
import numpy as np
from PIL import Image


class Frame:
    def __init__(self, encodedFrame):
        self.encodedFrame = encodedFrame
        self.decodedFrame = None
        self.detectedWasteTypes = []

    def decode_image(self):
        img_bytes = base64.b64decode(self.encodedFrame)
        pimg = Image.open(io.BytesIO(img_bytes))
        self.decodedFrame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)