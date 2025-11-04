import onnxruntime as ort
import numpy as np
import cv2


class StickDetector:
    """Детектор дерева и палки (ствола) с помощью YOLOv8-ONNX."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print(f"📦 StickDetector загружен: {model_path}")

    def preprocess(self, image):
        """Resize + normalize под YOLOv8."""
        img = cv2.resize(image, (640, 640))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)
        img = img.astype(np.float32) / 255.0
        return img

    def detect(self, image):
        """Возвращает высоту и диаметр дерева."""
        try:
            input_tensor = self.preprocess(image)
            input_name = self.session.get_inputs()[0].name
            preds = self.session.run(None, {input_name: input_tensor})[0]

            # Для YOLOv8: [x, y, w, h, conf, class]
            boxes = []
            for det in preds[0]:
                x, y, w, h, conf, cls = det[:6]
                if conf < 0.5:
                    continue
                boxes.append((x, y, w, h, conf, int(cls)))

            if not boxes:
                print("⚠️ Объекты не найдены.")
                return None

            # Сортировка по размеру (первое — дерево)
            boxes.sort(key=lambda b: b[3], reverse=True)
            tree_box = boxes[0]
            x, y, w, h, conf, cls = tree_box

            # Пример расчётов (условно)
            height_m = (h / 640) * 20.0
            diameter_cm = (w / 640) * 100.0

            return {
                "bbox": [int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)],
                "height": round(height_m, 2),
                "diameter": round(diameter_cm, 1),
                "confidence": float(conf)
            }

        except Exception as e:
            print(f"❌ Ошибка в StickDetector: {e}")
            return None
