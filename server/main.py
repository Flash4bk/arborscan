import io
import os
import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, get_soil, compute_risk
from server.stick_detector import StickDetector

app = FastAPI(title="ArborScan API", version="2.5")

# --- Загрузка моделей ---
print("🔄 Загружаем модели ONNX...")
TREE_SEG_MODEL = "server/models/tree_seg.onnx"
tree_sess = ort.InferenceSession(TREE_SEG_MODEL, providers=["CPUExecutionProvider"])
stick_detector = StickDetector("server/models/stick_yolo.onnx")
print("✅ Модели успешно загружены.")


@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(None),
    lon: float = Form(None)
):
    """Основной эндпоинт анализа дерева (с сегментацией по маске)."""
    try:
        # === 1. Чтение изображения ===
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Ошибка: не удалось декодировать изображение.")
        h, w = image.shape[:2]
        print(f"📷 Изображение получено: {w}x{h}")

        # === 2. Классификация дерева ===
        species, confidence = classify_tree(image)
        print(f"🌿 Определён вид: {species} ({confidence * 100:.1f}% уверенности)")

        # === 3. Сегментация дерева ===
        seg_input = cv2.resize(image, (640, 640))
        seg_input = cv2.cvtColor(seg_input, cv2.COLOR_BGR2RGB)
        seg_input = seg_input.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        mask_pred = tree_sess.run(None, {tree_sess.get_inputs()[0].name: seg_input})[0]
        mask_pred = mask_pred[0][0] if mask_pred.ndim == 4 else mask_pred[0]
        mask_resized = cv2.resize(mask_pred, (w, h))
        mask_bin = (mask_resized > 0.35).astype(np.uint8)
        print(f"🟢 Маска дерева получена: {mask_bin.shape}, тип={mask_bin.dtype}")

        # === 4. Детекция ствола / палки ===
        detections = stick_detector.detect(image)
        if not detections:
            print("⚠️ Палка не найдена, масштаб неизвестен.")
        else:
            print(f"📏 Высота={detections['height']:.2f}м, D={detections['diameter']:.1f}см")

        # === 5. Погода и почва ===
        if lat and lon:
            try:
                weather = get_weather(lat, lon)
                soil = get_soil(lat, lon)
            except Exception:
                weather, soil = None, None
                print("⚠️ Ошибка при получении данных погоды или почвы.")
        else:
            weather, soil = None, None
            print("⚠️ Нет данных GPS, пропущены погодные параметры.")

        # === 6. Риск падения ===
        try:
            risk_level, risk_score = compute_risk(detections or {}, weather, soil)
        except Exception as e:
            print(f"⚠️ Ошибка при расчёте риска: {e}")
            risk_level, risk_score = "Не рассчитано", 0.0

        # === 7. Визуализация ===
        vis = image.copy()

        # --- Маска дерева ---
        if np.sum(mask_bin) > 0:
            colored_mask = np.zeros_like(vis)
            colored_mask[:, :, 1] = mask_bin * 255  # зелёный канал
            vis = cv2.addWeighted(vis, 1, colored_mask, 0.4, 0)
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 255, 0), 3)
            print("✅ Контуры дерева отрисованы.")

        # --- Прямоугольник дерева ---
        if detections and "bbox" in detections:
            x1, y1, x2, y2 = map(int, detections["bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"H={detections['height']:.1f}m",
                        (x1, max(y1 - 10, 15)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            cv2.putText(vis, f"D={detections['diameter']:.1f}cm",
                        (x1, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)

        # --- Палка ---
        if detections and "sticks" in detections:
            for stick_box in detections["sticks"]:
                sx1, sy1, sx2, sy2 = map(int, stick_box)
                cv2.rectangle(vis, (sx1, sy1), (sx2, sy2), (0, 0, 255), 2)
                cv2.putText(vis, "stick", (sx1, max(sy1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Сохранение изображения ---
        os.makedirs("server/output", exist_ok=True)
        output_path = "server/output/analyzed_tree.png"
        cv2.imwrite(output_path, vis)
        print(f"📸 Визуализация сохранена: {output_path}")

        # === 8. Финальный ответ ===
        # Приведение всех значений к стандартным типам
        def safe(v):
            """Приводит numpy.float и др. к стандартному Python float."""
            if isinstance(v, (np.floating, np.integer)):
                return float(v)
            return v

        result = {
            "species": str(species),
            "confidence": safe(confidence),
            "geometry": {
                "height_m": safe(detections["height"]) if detections else None,
                "diameter_cm": safe(detections["diameter"]) if detections else None
            },
            "risk": {
                "level": str(risk_level),
                "score": safe(risk_score)
            },
            "weather": weather if weather else "Нет данных",
            "soil": soil if soil else "Нет данных",
            "visualization_path": "server/output/analyzed_tree.png"
        }

        print("✅ Анализ завершён успешно.")
        return JSONResponse(content=result)

    except Exception as e:
        print(f"❌ Ошибка при анализе: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})



@app.get("/")
async def root():
    return {"status": "ok", "message": "ArborScan API online"}
