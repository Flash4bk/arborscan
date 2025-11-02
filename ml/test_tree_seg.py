import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import os

# === Пути ===
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "onnx", "tree_seg.onnx")

# === Укажи путь к фото дерева ===
IMAGE_PATH = os.path.join(BASE_DIR, "test_tree.jpg")  # <-- положи сюда любое дерево

# === Загружаем изображение ===
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"Не найдено изображение: {IMAGE_PATH}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === Препроцессинг для YOLOv8-seg ===
# Размер входа 640x640
img_resized = cv2.resize(img_rgb, (640, 640))
img_norm = img_resized.astype(np.float32) / 255.0
img_input = np.transpose(img_norm, (2, 0, 1))  # HWC -> CHW
img_input = np.expand_dims(img_input, axis=0)  # batch dim

# === Загружаем ONNX модель ===
sess = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
output_names = [o.name for o in sess.get_outputs()]
print("✅ Model loaded:", MODEL_PATH)
print("Inputs:", input_name)
print("Outputs:", output_names)

# === Инференс ===
res = sess.run(output_names, {input_name: img_input})

# === Для YOLOv8-seg обычно 2 выхода: детекция и маски ===
for i, r in enumerate(res):
    print(f"Output[{i}]:", r.shape)

# Обычно res[0] → [1, N, 84] (bbox + cls + mask coeffs)
#         res[1] → [1, 32, 160, 160] (mask protos)

# === Простая визуализация одной маски (если модель сегментационная) ===
if len(res) == 2 and res[1].ndim == 4:
    protos = res[1][0]  # [C, H, W]
    # Маски вычисляются как линейная комбинация прототипов * coeffs
    # Возьмем просто среднее для примера визуализации
    mask = protos.mean(axis=0)
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-5)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # === Отрисовка маски поверх исходного изображения ===
    overlay = img_rgb.copy().astype(np.float32)
    overlay[:, :, 1] += mask * 180  # усилим зелёный канал
    overlay = np.clip(overlay / overlay.max(), 0, 1)

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.title("Исходное фото")
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Маска дерева (пример)")
    plt.imshow(overlay)
    plt.axis("off")
    plt.show()

else:
    print("⚠️ Не удалось распознать структуру выхода. Возможно, это не сегментационная модель YOLOv8-seg.")
