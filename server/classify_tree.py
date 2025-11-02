import onnxruntime as ort
import numpy as np
from PIL import Image
import cv2

labels = ["Берёза", "Дуб", "Ель", "Сосна", "Тополь"]
CLASSIFIER_MODEL = "server/models/classifier.onnx"

def classify_tree(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((224, 224))
    img = np.asarray(img, dtype=np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]

    sess = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])
    probs = sess.run(None, {"image": img})[0][0]
    pred = np.argmax(probs)
    confidence = float(np.max(probs))
    return labels[pred], confidence
