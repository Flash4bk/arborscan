import os
import io
import cv2
import base64
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

# Внутренние импорты
from classify_tree import classify_tree
from stick_detector import StickDetector
from risk_analysis import compute_risk

# --- Инициализация FastAPI ---
app = FastAPI(title="ArborScan Server")

# --- Загружаем модели ---
print("🔄 Загружаем модели ONNX...")
stick_detector = StickDetector("server/models/stick_yolo.onnx")
print("✅ StickDetector загружен:", stick_detector.session.get_inputs()[0].shape)
print("✅ Модели успешно загружены.")


# --- Вспомогательная функция для преобразования изображения ---
def read_imagefile(file) -> np.ndarray:
    image = Image.open(io.BytesIO(file))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


# --- Основной маршрут анализа ---
@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(None),
    lon: float = Form(None)
):
    try:
        # Чтение и сохранение файла
        contents = await file.read()
        image = read_imagefile(contents)
        height, width, _ = image.shape
        print(f"📷 Изображение получено: {width}x{height}")

        # Классификация дерева
        species, conf = classify_tree("server/temp.jpg")
        print(f"🌿 Определён вид: {species} ({conf:.1f}% уверенности)")

        # Детекция дерева и палки
        tree_mask = np.zeros((height, width), dtype=np.uint8)
        tree_mask[:height // 2, :] = 255  # временно, если нет сегментации

        stick_mask = None
        stick_height = None
        try:
            stick_mask, stick_height = stick_detector.detect_stick(image)
        except Exception as e:
            print(f"⚠️ Ошибка StickDetector: {e}")
            stick_height = None

        # Параметры дерева
        H = round(height / 100, 2)
        D = round(width / 100, 2)
        print(f"📏 Высота={H}м, Диаметр={D}см")

        # Если нет GPS — предупреждение
        if lat is None or lon is None:
            print("⚠️ Нет данных GPS, пропущен анализ погоды и почвы.")
            weather = {"wind": 0, "gust": 0, "temp": 0}
            soil = None
        else:
            # Здесь может быть получение данных погоды/почвы
            weather = {"wind": 2.3, "gust": 5.6, "temp": 6.4}
            soil = {"sand": 30, "clay": 20, "organic": 2.1}

        # Риск падения
        risk, risk_level = compute_risk(H, D, weather["wind"], soil)
        print(f"⚖️ Риск падения: {risk_level} ({risk:.1f}/100)")

        # --- Визуализация ---
        vis = image.copy()
        if stick_mask is not None:
            vis[stick_mask > 0] = [0, 0, 255]
        cv2.putText(vis, f"H={H}m", (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(vis, f"D={D}cm", (25, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Сохраняем визуализацию
        os.makedirs("server/output", exist_ok=True)
        out_path = "server/output/analyzed_tree.png"
        cv2.imwrite(out_path, vis)
        print(f"🖼️ Визуализация сохранена: {out_path}")

        # Кодируем изображение в base64
        _, buffer = cv2.imencode(".png", vis)
        img_base64 = base64.b64encode(buffer).decode("utf-8")

        # --- Ответ клиенту ---
        response = {
            "species": species,
            "confidence": conf,
            "height_m": H,
            "diameter_cm": D,
            "weather": weather,
            "risk_level": risk_level,
            "risk_score": risk,
            "image_base64": img_base64
        }

        print("✅ Анализ завершён успешно.")
        return JSONResponse(content=response)

    except Exception as e:
        print(f"❌ Ошибка при анализе изображения: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
