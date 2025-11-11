import onnxruntime as ort
import numpy as np
import cv2


class StickDetector:
    def __init__(self, model_path="server/models/stick_yolo.onnx"):
        """Инициализация модели для детекции палки"""
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Подготовка изображения"""
        img = cv2.resize(image, (768, 768))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def detect_stick(self, image: np.ndarray):
        """Поиск палки на фото"""
        blob = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: blob})
        detections = outputs[0]

        # Находим самую крупную детекцию (палку)
        if len(detections) == 0:
            raise ValueError("Палка не найдена")

        best_det = detections[0]
        x1, y1, x2, y2 = map(int, best_det[:4])
        conf = float(best_det[4])

        if conf < 0.3:
            raise ValueError("Слишком низкая уверенность детекции палки")

        # Создаём маску
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        stick_height_m = round(abs(y2 - y1) / 100, 2)
        print(f"📏 Палки найдена: высота ≈ {stick_height_m} м")

        return mask, stick_height_m
