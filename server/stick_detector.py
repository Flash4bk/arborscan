import cv2
import numpy as np
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)

class StickDetector:
    def __init__(self, model_path: str):
        self.model_path = model_path
        logger.info(f"Загрузка модели палки из {model_path}...")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        logger.info(f"StickDetector загружен: {self.input_shape}")

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Подготовка изображения"""
        h, w = self.input_shape[2], self.input_shape[3]
        img = cv2.resize(image, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def detect_stick(self, image: np.ndarray):
        """Детектирование палки"""
        try:
            input_tensor = self.preprocess(image)
            outputs = self.session.run(None, {self.input_name: input_tensor})

            if len(outputs) == 0 or outputs[0] is None:
                logger.warning("Выход модели пустой — палка не найдена.")
                return None

            detections = np.array(outputs[0])
            if detections.size == 0:
                logger.warning("Детекции отсутствуют.")
                return None

            best_det = detections[0]

            # Безопасно извлекаем значения
            values = []
            for v in best_det[:5]:
                if isinstance(v, np.ndarray):
                    v = v.flatten()
                    if v.size > 0:
                        v = v[0]
                try:
                    values.append(float(v))
                except Exception:
                    values.append(0.0)

            x1, y1, x2, y2, conf = values
            logger.info(f"Палка найдена: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}, conf={conf:.2f}")

            return (int(x1), int(y1), int(x2), int(y2), conf)

        except Exception as e:
            logger.error(f"Ошибка StickDetector: {e}")
            return None
