import os
import cv2
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from classify_tree import classify_tree
from risk_analysis import get_weather, get_soil, soil_factor, compute_risk

# === Пути ===
BASE_DIR = os.path.dirname(__file__)
TREE_MODEL = os.path.join(BASE_DIR, "onnx", "tree_seg.onnx")
STICK_MODEL = os.path.join(BASE_DIR, "onnx", "stick_yolo.onnx")
CLASSIFIER_MODEL = os.path.join(BASE_DIR, "onnx", "classifier.onnx")
IMAGE_PATH = os.path.join(BASE_DIR, "test_tree.jpg")

# === 1. Определение длины палки 1 м (YOLOv8) ===
def detect_stick_length(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    img = cv2.resize(img_bgr, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))[None, :, :, :]

    sess = ort.InferenceSession(STICK_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = [o.name for o in sess.get_outputs()]
    res = sess.run(outputs, {input_name: img})

    det = res[0][0]  # [N, 6]: x1,y1,x2,y2,conf,class
    if det.shape[0] == 0:
        print("⚠️ Палка не найдена, масштаб = 1/200 по умолчанию")
        return 200

    # Берем самую уверенную детекцию
    best = det[np.argmax(det[:, 4])]
    x1, y1, x2, y2 = best[:4]
    length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    scale_factor = h0 / 640  # коррекция под исходное изображение
    return length_px * scale_factor

# === 2. Сегментация дерева (YOLOv8-seg) ===
def segment_tree_mask(img_bgr):
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(img_rgb, (640, 640)).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]

    sess = ort.InferenceSession(TREE_MODEL, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = [o.name for o in sess.get_outputs()]
    res = sess.run(outputs, {input_name: inp})

    # Извлекаем mask protos
    protos = res[1][0] if len(res) > 1 else None
    if protos is None:
        raise RuntimeError("Модель сегментации не вернула маску.")
    mask = protos.mean(axis=0)
    mask = cv2.resize(mask, (w0, h0))
    mask_bin = (mask > np.percentile(mask, 85)).astype(np.uint8)
    return mask_bin

# === 3. Основная функция измерений ===
def analyze_tree():
    img_bgr = cv2.imread(IMAGE_PATH)
    if img_bgr is None:
        raise FileNotFoundError("❌ Изображение test_tree.jpg не найдено в папке ml/")

    # --- Сегментация дерева ---
    mask_bin = segment_tree_mask(img_bgr)

    # --- Определение длины эталона ---
    Lpx = detect_stick_length(img_bgr)
    scale = 1.0 / Lpx
    print(f"Палка найдена, длина = {Lpx:.1f}px, масштаб = {scale:.6f} м/пикс")

    # --- Геометрические измерения ---
    ys, xs = np.where(mask_bin > 0)
    y_top, y_bottom = ys.min(), ys.max()
    H_m = (y_bottom - y_top) * scale  # высота

    # DBH (диаметр на уровне груди 1.3 м)
    y_dbh = int(y_bottom - 1.3 / scale)
    row = mask_bin[y_dbh, :] if 0 <= y_dbh < mask_bin.shape[0] else np.zeros_like(mask_bin[0])
    if np.any(row):
        x_left, x_right = np.where(row > 0)[0][[0, -1]]
        dbh_px = x_right - x_left
        DBH_m = dbh_px * scale
    else:
        DBH_m = 0

    # Длина кроны
    widths = np.array([mask_bin[y, :].sum() for y in range(y_top, y_bottom)], dtype=np.float32)
    dy = np.gradient(widths)
    if widths.size > 0:
        thresh = widths.max() * 0.3
        if np.any(dy > thresh):
            crown_base_rel = int(np.argmax(dy > thresh))
        else:
            crown_base_rel = int(len(widths) * 0.6)
    else:
        crown_base_rel = 0
    CL_px = (y_bottom - (y_top + crown_base_rel))
    CL_m = CL_px * scale

    print(f"Высота дерева: {H_m:.2f} м")
    print(f"Длина кроны: {CL_m:.2f} м")
    print(f"Диаметр ствола (DBH): {DBH_m*100:.1f} см")

    # --- Визуализация маски ---
    overlay = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    overlay[mask_bin > 0] = [0, 255, 0]
    plt.imshow(overlay)
    plt.title("Маска и измерения")
    plt.axis("off")
    plt.show()

    return H_m, CL_m, DBH_m

# === 4. Основной блок выполнения ===
if __name__ == "__main__":
    H_m, CL_m, DBH_m = analyze_tree()

    # --- Получаем вид дерева ---
    species, confidence = classify_tree(IMAGE_PATH)
    print(f"🌿 Определён вид: {species} ({confidence*100:.1f}% уверенности)")

    # --- Координаты (пример: Москва) ---
    lat, lon = 55.75, 37.62

    # --- Погода и почва ---
    wind_speed, gust, temp = get_weather(lat, lon)
    clay, sand, silt, bd, oc = get_soil(lat, lon)
    k_soil = soil_factor(clay, sand)

    # --- Расчёт риска ---
    risk, level = compute_risk(species, H_m, DBH_m, CL_m, wind_speed, gust, k_soil)

    # --- Вывод ---
    print(f"\n🌳 Вид: {species}")
    print(f"Высота: {H_m:.1f} м, Крона: {CL_m:.1f} м, Ствол: {DBH_m*100:.1f} см")
    print(f"Почва: k={k_soil:.2f}, Ветер={wind_speed} м/с, Порывы={gust} м/с")
    print(f"👉 Риск падения: {level} ({risk:.1f}/100)")
