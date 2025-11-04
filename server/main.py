import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, get_soil, compute_risk
from server.stick_detector import StickDetector

app = FastAPI(title="ArborScan API", version="2.0")

# Инициализация моделей при старте
print("🔄 Загружаем модели ONNX...")
stick_model = StickDetector("server/models/stick_yolo.onnx")
print("✅ Модели успешно загружены.")


@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(None),
    lon: float = Form(None)
):
    """Основной эндпоинт анализа дерева."""
    try:
        # Читаем изображение
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        print(f"📷 Изображение получено: {image.shape[1]}x{image.shape[0]}")

        # Классификация
        species, confidence = classify_tree(image)
        print(f"🌿 Определён вид: {species} ({confidence * 100:.1f}% уверенности)")

        # Анализ с помощью детектора
        tree_data = stick_model.detect(image)
        print(f"📏 Высота={tree_data['height']:.2f}м, D={tree_data['diameter']:.1f}см")

        # Получаем метео-данные
        if lat and lon:
            weather = get_weather(lat, lon)
            soil = get_soil(lat, lon)
            risk = compute_risk(tree_data, weather, soil)
        else:
            weather, soil, risk = None, None, "Нет GPS-данных"

        result = {
            "species": species,
            "confidence": confidence,
            "geometry": tree_data,
            "weather": weather,
            "soil": soil,
            "risk": risk,
        }

        print("✅ Анализ завершён успешно.")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ Ошибка при анализе изображения: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
