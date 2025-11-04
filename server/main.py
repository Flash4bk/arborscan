import io
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, get_soil, compute_risk
from server.stick_detector import StickDetector
import os

app = FastAPI(title="ArborScan API", version="2.3")

print("🔄 Загружаем модели ONNX...")
stick_detector = StickDetector("server/models/stick_yolo.onnx")
print("✅ Модели успешно загружены.")


@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(None),
    lon: float = Form(None)
):
    """Основной эндпоинт анализа дерева."""
    try:
        # --- 1. Чтение изображения ---
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Ошибка: не удалось декодировать изображение.")
        h, w = image.shape[:2]
        print(f"📷 Изображение получено: {w}x{h}")

        # --- 2. Классификация дерева ---
        species, confidence = classify_tree(image)
        print(f"🌿 Определён вид: {species} ({confidence * 100:.1f}% уверенности)")

        # --- 3. Детекция ствола и палки ---
        tree_data = stick_detector.detect(image)
        if not tree_data:
            raise ValueError("Не удалось определить дерево или ствол.")
        print(f"📏 Высота={tree_data['height']:.2f}м, D={tree_data['diameter']:.1f}см")

        # --- 4. Получение данных о погоде и почве ---
        if lat and lon:
            try:
                weather = get_weather(lat, lon)
                soil = get_soil(lat, lon)
            except Exception:
                weather, soil = None, None
                print("⚠️ Ошибка при получении данных почвы или погоды.")
        else:
            weather, soil = None, None
            print("⚠️ Нет данных GPS, пропущены почва и погода.")

        # --- 5. Расчёт риска ---
        try:
            risk_level, risk_score = compute_risk(tree_data, weather, soil)
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте риска: {e}")
            risk_level, risk_score = "Не рассчитано", 0.0

        # --- 6. Визуализация ---
        vis_image = image.copy()
        if "bbox" in tree_data:
            x1, y1, x2, y2 = map(int, tree_data["bbox"])
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_image, f"H={tree_data['height']:.1f}m",
                        (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(vis_image, f"D={tree_data['diameter']:.1f}cm",
                        (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        os.makedirs("server/output", exist_ok=True)
        output_path = "server/output/analyzed_tree.png"
        cv2.imwrite(output_path, vis_image)
        print(f"📸 Визуализация сохранена: {output_path}")

        # --- 7. Ответ API ---
        result = {
            "species": species,
            "confidence": confidence,
            "geometry": {
                "height_m": tree_data["height"],
                "diameter_cm": tree_data["diameter"]
            },
            "risk": {
                "level": risk_level,
                "score": risk_score
            },
            "weather": weather if weather else "Нет данных (без GPS)",
            "soil": soil if soil else "Нет данных (без GPS)",
            "visualization_path": output_path
        }

        print("✅ Анализ завершён успешно.")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ Ошибка при анализе изображения: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/")
async def root():
    return {"status": "ok", "message": "ArborScan API online"}
