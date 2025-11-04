import numpy as np
import cv2
import onnxruntime as ort


# Загружаем модель один раз при инициализации
MODEL_PATH = "server/models/classifier.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Маппинг меток классов (пример)
TREE_LABELS = ["Берёза", "Сосна", "Ель", "Дуб", "Тополь", "Неизвестно"]


def preprocess_image(image: np.ndarray):
    """Предобработка изображения под модель классификации."""
    img = cv2.resize(image, (224, 224))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # (HWC → CHW)
    img = np.expand_dims(img, axis=0)   # Добавляем batch
    return img


def classify_tree(image: np.ndarray, model=None):
    """
    Классификация дерева.
    image — numpy.ndarray
    model — (необязательно) объект модели ONNX
    """
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        img_input = preprocess_image(image)
        pred = session.run([output_name], {input_name: img_input})[0]

        idx = int(np.argmax(pred))
        confidence = float(pred[0][idx])

        label = TREE_LABELS[idx] if idx < len(TREE_LABELS) else "Неизвестно"

        return label, confidence

    except Exception as e:
        print(f"Ошибка классификации дерева: {e}")
        return "Неизвестно", 0.0
