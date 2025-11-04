import onnxruntime as ort
import numpy as np
from PIL import Image

labels = ["Берёза", "Дуб", "Ель", "Сосна", "Тополь"]
CLASSIFIER_MODEL = "server/models/classifier.onnx"


def classify_tree(image_input):
    """
    Классификация дерева по изображению.
    Принимает либо путь к файлу, либо np.ndarray (RGB).
    """
    try:
        # --- если пришёл путь к файлу ---
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
            img = np.asarray(img, dtype=np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None, :, :, :]

        # --- если пришёл numpy-массив ---
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 3 and image_input.shape[2] == 3:
                img = image_input.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))[None, :, :, :]
            else:
                raise ValueError("Некорректный формат изображения (ожидалось RGB).")

        else:
            raise TypeError("Передан неподдерживаемый тип изображения")

        # --- классификация ---
        sess = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])
        probs = sess.run(None, {"image": img})[0][0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100.0
        return labels[pred], confidence

    except Exception as e:
        print("Ошибка в classify_tree:", e)
        return "Неизвестно", 0.0
