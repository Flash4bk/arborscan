import onnxruntime as ort
import numpy as np
import cv2

labels = ["Берёза", "Дуб", "Ель", "Сосна", "Тополь"]
CLASSIFIER_MODEL = "server/models/classifier.onnx"


def classify_tree(image_input):
    """
    Классификация дерева по фото.
    Поддерживает np.ndarray RGB (например, из OpenCV или PIL).
    """
    try:
        # --- если пришёл NumPy массив ---
        if isinstance(image_input, np.ndarray):
            # приводим к RGB (если вдруг BGR)
            if image_input.shape[2] == 3:
                img_rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Ожидалось изображение с 3 каналами (RGB).")

            # ресайз под ResNet-18
            img_resized = cv2.resize(img_rgb, (224, 224))
            img = img_resized.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))[None, :, :, :]  # [1,3,224,224]

        else:
            raise TypeError("classify_tree принимает только np.ndarray")

        # --- инференс ---
        sess = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])
        probs = sess.run(None, {"image": img})[0][0]
        pred = int(np.argmax(probs))
        conf = float(np.max(probs)) * 100
        return labels[pred], conf

    except Exception as e:
        print("Ошибка в classify_tree:", e)
        return "Неизвестно", 0.0
