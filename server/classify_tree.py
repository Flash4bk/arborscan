import onnxruntime as ort
import numpy as np
import cv2

CLASSIFIER_MODEL = "server/models/classifier.onnx"

# Инициализация ONNX-сессии
sess = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])

# Метки классов (замени своими)
TREE_CLASSES = ["Берёза", "Дуб", "Ель", "Сосна", "Тополь"]


def classify_tree(image_path: str):
    """Классификация дерева по изображению"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось открыть изображение: {image_path}")

    # Подготовка входных данных (Resize + Normalize)
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    pred = sess.run([output_name], {input_name: img})[0]
    probs = np.exp(pred) / np.sum(np.exp(pred))
    idx = int(np.argmax(probs))
    conf = float(probs[0][idx]) * 100

    return TREE_CLASSES[idx], round(conf, 1)
