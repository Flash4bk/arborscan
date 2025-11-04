import os
import io
import cv2
import math
import json
import requests
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, compute_risk

app = FastAPI(title="ArborScan API", description="Tree analysis server")

# Разрешаем обращения от Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пути к моделям
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_MODEL = os.path.join(MODEL_DIR, "classifier.onnx")
STICK_MODEL = os.path.join(MODEL_DIR, "stick_yolo.onnx")
TREE_MODEL = os.path.join(MODEL_DIR, "tree_seg.onnx")

print("🔄 Загружаем модели...")
sess_classifier = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])
sess_stick = ort.InferenceSession(STICK_MODEL, providers=["CPUExecutionProvider"])
sess_tree = ort.InferenceSession(TREE_MODEL, providers=["CPUExecutionProvider"])
print("✅ Модели загружены успешно.")


# ========================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ==========================

def detect_stick(image_rgb: np.ndarray):
    """Находим эталонную палку (1 м)"""
    img_resized = cv2.resize(image_rgb, (640, 640))
    inp = img_resized.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    out = sess_stick.run(None, {sess_stick.get_inputs()[0].name: inp})
    det = out[0][0]
    if det.shape[0] == 0:
        return None

    best = det[np.argmax(det[:, 4])]
    x1, y1, x2, y2, conf, *_ = best
    length_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print(f"📏 Палка найдена (длина {length_px:.1f}px, conf={conf:.2f})")
    return length_px


def segment_tree(image_rgb: np.ndarray):
    """Сегментация дерева"""
    img_resized = cv2.resize(image_rgb, (640, 640))
    inp = img_resized.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]
    out = sess_tree.run(None, {sess_tree.get_inputs()[0].name: inp})
    mask = out[1][0] if len(out) > 1 else out[0][0]
    mask_bin = (mask > 0.35).astype(np.uint8)
    kernel = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel)
    return mask_bin


# =============================== ГЛАВНЫЙ ЭНДПОИНТ ===============================

@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(None),
    lon: float = Form(None),
    api_key: str = Form(""),
):
    try:
        print("📸 Анализ изображения...")

        # --- 1️⃣ читаем байты и открываем через PIL ---
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # --- 2️⃣ конвертируем в numpy (OpenCV формат RGB) ---
        img = np.array(image)
        h, w = img.shape[:2]
        print(f"Изображение получено: {w}x{h}")

        # --- 3️⃣ продолжаем как обычно ---
        species, conf = classify_tree(img)
        print(f"🌿 Вид дерева: {species} ({conf:.1f}%)")


        # 3️⃣ Поиск эталонной палки
        stick_len_px = detect_stick(img)
        if not stick_len_px:
            return JSONResponse({"error": "Палка не найдена на фото."}, status_code=400)

        scale = 1.0 / stick_len_px  # 1 метр палки

        # 4️⃣ Сегментация дерева
        mask_bin = segment_tree(img)
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return JSONResponse({"error": "Не удалось выделить дерево."}, status_code=400)
        cnt = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        ys, xs = np.where(mask_bin > 0)
        y_bottom = np.max(ys)
        y_top = int(np.percentile(ys, 5))

        # 5️⃣ Геометрия
        height_px = y_bottom - y_top
        height_m = height_px * scale
        row_y = int(y_bottom - height_px * 0.05)
        xs_row = np.where(mask_bin[row_y, :] > 0)[0]
        dbh_px = (xs_row[-1] - xs_row[0]) if len(xs_row) > 1 else 0
        dbh_m = dbh_px * scale
        crown_m = height_m / 3.0

        # 6️⃣ Погода
        wind, gust, temp = (None, None, None)
        if lat and lon and api_key:
            wind, gust, temp = get_weather(lat, lon, api_key)

        # 7️⃣ Риск (пока пустой — для совместимости клиента)
        risk = {
            "level": None,
            "score": None,
            "reasons": [],
            "note": "Риск пока не рассчитан (нет данных почвы/ветра)."
        }

        # 8️⃣ Визуализация
        vis = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        color_mask = np.zeros_like(vis)
        color_mask[:, :, 1] = (mask_bin * 255)
        vis = cv2.addWeighted(vis, 0.8, color_mask, 0.3, 0)
        cv2.rectangle(vis, (x, y_top), (x + w_box, y_bottom), (255, 0, 0), 2)
        cv2.putText(vis, f"H={height_m:.1f}m", (x+5, y_top+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"D={dbh_m*100:.1f}cm", (x+5, y_bottom-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = os.path.join(BASE_DIR, "analyzed_tree.png")
        cv2.imwrite(out_path, vis)

        # 9️⃣ Формирование ответа
        result = {
            "species": species,
            "confidence": conf,
            "height_m": round(height_m, 2),
            "crown_m": round(crown_m, 2),
            "dbh_cm": round(dbh_m * 100, 1),
            "wind": wind,
            "gust": gust,
            "temperature": temp,
            "risk": risk,
            "image_path": "/image"
        }

        print("✅ Анализ завершён успешно.")
        return JSONResponse(result)

    except Exception as e:
        print("❌ Ошибка:", e)
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/image")
def get_image():
    path = os.path.join(BASE_DIR, "analyzed_tree.png")
    if not os.path.exists(path):
        return JSONResponse({"error": "Нет изображения анализа."}, status_code=404)
    return FileResponse(path, media_type="image/png", filename="analyzed_tree.png")
