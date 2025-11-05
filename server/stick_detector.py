import cv2
import numpy as np
import onnxruntime as ort


class StickDetector:
    def __init__(self, model_path: str):
        print(f"🔄 Загружаем StickDetector из {model_path}...")
        self.model = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.model.get_inputs()[0].name
        input_shape = self.model.get_inputs()[0].shape
        self.input_size = (input_shape[3], input_shape[2]) if len(input_shape) == 4 else (640, 640)
        print(f"✅ StickDetector готов: вход модели {self.input_size}")

    def detect(self, image: np.ndarray):
        """Детектирует дерево и возвращает маску и геометрию."""
        img = cv2.resize(image, self.input_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_in = img_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_in = np.expand_dims(img_in, axis=0)

        outputs = self.model.run(None, {self.input_name: img_in})
        mask = (outputs[0][0] > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        # Вычисляем параметры
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {}, mask
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        height_m = round(h / 100, 2)
        diameter_cm = round(w / 5, 2)

        return {"height": height_m, "diameter": diameter_cm}, mask

    def draw_detections(self, image, detections, mask):
        """Рисует результат анализа."""
        vis = image.copy()
        if mask is not None:
            colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_SUMMER)
            vis = cv2.addWeighted(vis, 0.7, colored_mask, 0.3, 0)

        x = 20
        y = 40
        cv2.rectangle(vis, (x, y - 30), (x + 180, y + 40), (255, 255, 255), -1)
        cv2.putText(vis, f"H={detections.get('height', 0)}m", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(vis, f"D={detections.get('diameter', 0)}cm", (x + 10, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        return vis
