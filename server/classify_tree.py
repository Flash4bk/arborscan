import onnxruntime as ort
import numpy as np
import cv2
import os

BASE_DIR = os.path.dirname(__file__)
CLASSIFIER_MODEL = os.path.join(os.path.dirname(__file__), "..", "server", "models", "classifier.onnx")
CLASSIFIER_MODEL = os.path.abspath(CLASSIFIER_MODEL)


CLASSES = ["Берёза", "Дуб", "Ель", "Сосна", "Тополь"]

def classify_tree(image_path):
    # === загрузка и препроцесс ===
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не найдено изображение: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]

    # === инференс ===
    sess = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = [o.name for o in sess.get_outputs()]
    probs = sess.run(outputs, {input_name: img})[0][0]

    probs = np.exp(probs) / np.sum(np.exp(probs))  # softmax, если нужно
    idx = int(np.argmax(probs))
    return CLASSES[idx], float(probs[idx])
