import os
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from io import BytesIO
from PIL import Image

from stick_detector import StickDetector
from classify_tree import classify_tree
from risk_analysis import get_weather, get_soil, compute_risk

# ---------------- ПУТИ -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TREE_MODEL = os.path.join(BASE_DIR, "models", "tree_seg.onnx")
CLASSIFIER_MODEL = os.path.join(BASE_DIR, "models", "classifier.onnx")
STICK_MODEL = os.path.join(BASE_DIR, "models", "stick_yolo.onnx")

# ---------------- СЕРВЕР -----------------
app = FastAPI(title="ArborScan API", version="2.1")

# ---------------- МОДЕЛИ -----------------
print("🚀 Загружаем модели ONNX...")
tree_session = ort.InferenceSession(TREE_MODEL, providers=["CPUExecutionProvider"])
class_session = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])
stick_detector = StickDetector(STICK_MODEL, img_size=640, conf_thres=0.15, iou_thres=0.45)
print("✅ Модели успешно загружены.")


# =========================================================
#                     ОСНОВНАЯ ФУНКЦИЯ
# =========================================================
@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(None),
    lon: float = Form(None)
):
    try:
        # ---------- Загружаем фото ----------
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        h_img, w_img = img_bgr.shape[:2]
        print(f"📷 Изображение получено: {w_img}x{h_img}")

        # ---------- Классификация вида ----------
        species, conf_cls = classify_tree(img_bgr, class_session)
        print(f"🌿 Определён вид: {species} ({conf_cls:.1f}% уверенности)")

        # ---------- Сегментация дерева ----------
        blob = cv2.resize(img_bgr, (640, 640))
        blob = blob.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        mask = tree_session.run(None, {tree_session.get_inputs()[0].name: blob})[0][0]
        mask = cv2.resize(mask, (w_img, h_img))
        mask_bin = (mask > 0.5).astype(np.uint8)

        # ---------- Поиск рейки ----------
        stick_box, stick_conf = stick_detector(img_bgr)
        scale_m_per_px = None
        overlay = img_bgr.copy()

        if stick_box is not None:
            x1, y1, x2, y2 = map(int, stick_box)
            stick_h_px = max(1, y2 - y1)
            scale_m_per_px = 1.0 / stick_h_px
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(overlay, f"Stick {stick_conf:.2f}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
            print(f"📏 Рейка найдена: {stick_h_px:.1f}px, conf={stick_conf:.2f}")
        else:
            print("⚠️ Рейка не найдена, масштаб неизвестен")

        # ---------- Извлекаем размеры дерева ----------
        y_indices = np.where(mask_bin.sum(axis=1) > 20)[0]
        if len(y_indices) > 0:
            y_top, y_bottom = y_indices[0], y_indices[-1]
            h_px = y_bottom - y_top
        else:
            h_px = 0

        # ---------- Вычисляем метрики ----------
        if scale_m_per_px is not None:
            height_m = h_px * scale_m_per_px
            diam_cm = 100 * 0.02 * h_px  # примерная пропорция DBH
        else:
            height_m = None
            diam_cm = None

        # ---------- Погода и почва ----------
        weather = get_weather(lat, lon) if lat and lon else None
        soil = get_soil(lat, lon) if lat and lon else None

        if weather is None:
            print("🌤 Нет данных GPS или погоды")
        if soil is None:
            print("🌍 Нет данных по почве")

        # ---------- Риск падения ----------
        if weather and soil and height_m:
            risk, level = compute_risk(height_m, diam_cm, weather, soil)
        else:
            risk, level = None, None

        # ---------- Визуализация ----------
        mask_colored = np.repeat(mask_bin[:, :, None], 3, axis=2) * np.array([0, 255, 0])
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored.astype(np.uint8), 0.3, 0)
        cv2.putText(overlay, f"H={height_m:.1f}m" if height_m else "H=?",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"D={diam_cm:.1f}cm" if diam_cm else "D=?",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        result_path = os.path.join(BASE_DIR, "analyzed_tree.png")
        cv2.imwrite(result_path, overlay)
        print(f"🖼 Визуализация сохранена: {result_path}")

        # ---------- Ответ ----------
        return JSONResponse({
            "species": species,
            "confidence": conf_cls,
            "height_m": round(height_m, 2) if height_m else None,
            "diameter_cm": round(diam_cm, 2) if diam_cm else None,
            "weather": weather if weather else "Нет данных",
            "soil": soil if soil else "Нет данных",
            "risk": risk if risk else "Не рассчитан",
            "risk_level": level if level else "Нет данных",
            "stick_detected": stick_conf is not None
        })

    except Exception as e:
        print(f"❌ Ошибка при анализе изображения: {e}")
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )
