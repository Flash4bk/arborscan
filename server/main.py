import io
import os
import cv2
import base64
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from classify_tree import classify_tree
from stick_detector import StickDetector
from risk_analysis import get_weather, get_soil, estimate_fall_risk

app = FastAPI(title="ArborScan Server")

# === 1. Инициализация моделей ===
print("🔄 Загружаем модели ONNX...")

CLASSIFIER_MODEL = "server/models/classifier.onnx"
STICK_MODEL_PATH = "server/models/stick_yolo.onnx"

stick_detector = StickDetector(STICK_MODEL_PATH)
print(f"✅ StickDetector загружен: {STICK_MODEL_PATH}")

print("✅ Модели успешно загружены.")


# === 2. Основной маршрут анализа ===
@app.post("/analyze")
async def analyze_tree(file: UploadFile = File(...), lat: float = Form(None), lon: float = Form(None)):
    try:
        # --- Сохраняем временное изображение ---
        image_bytes = await file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Ошибка при чтении изображения.")
        h, w, _ = img.shape
        print(f"📷 Изображение получено: {w}x{h}")

        # --- Классификация породы дерева ---
        species, confidence = classify_tree(img)
        print(f"🌿 Определён вид: {species} ({confidence:.1f}% уверенности)")

        # --- Сегментация дерева ---
        detections, mask = stick_detector.detect(img)
        if mask is not None:
            print(f"🌳 Маска дерева получена: {mask.shape}, тип={mask.dtype}")
        else:
            print("⚠️ Маска дерева не найдена.")

        # --- Погода и почва ---
        weather = None
        soil = None
        if lat and lon:
            weather = get_weather(lat, lon)
            soil = get_soil(lat, lon)
        else:
            print("⚠️ Нет данных GPS, пропущен погодный параметр.")

        # --- Геометрия дерева ---
        height = detections.get("height", 0)
        diameter = detections.get("diameter", 0)
        print(f"📏 Высота={height:.2f}м, Диаметр={diameter:.2f}см")

        # --- Оценка риска ---
        risk_level, risk_score = estimate_fall_risk(height, diameter, weather)
        print(f"⚠️ Риск падения: {risk_level}, {risk_score:.1f}/100")

        # --- Визуализация ---
        vis_img = stick_detector.draw_detections(img, detections, mask)
        os.makedirs("server/output", exist_ok=True)
        output_path = "server/output/analyzed_tree.png"
        cv2.imwrite(output_path, vis_img)
        print(f"🖼️ Визуализация сохранена: {output_path}")

        # --- Конвертация изображения в base64 ---
        image_base64 = None
        try:
            with open(output_path, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            print(f"⚠️ Не удалось закодировать изображение: {e}")

        # --- Формирование безопасного JSON ---
        def safe(v):
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v

        result = {
            "species": str(species),
            "confidence": safe(confidence),
            "geometry": {
                "height_m": safe(height),
                "diameter_cm": safe(diameter)
            },
            "risk": {
                "level": str(risk_level),
                "score": safe(risk_score)
            },
            "weather": weather if weather else "Нет данных",
            "soil": soil if soil else "Нет данных",
            "visualization_base64": image_base64
        }

        print("✅ Анализ завершён успешно.")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


# === 3. Проверка доступности сервера ===
@app.get("/")
async def root():
    return {"status": "ok", "message": "ArborScan backend работает корректно"}
