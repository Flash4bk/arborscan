import cv2
import numpy as np
import onnxruntime as ort
import math
import requests
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image
import os

app = FastAPI(title="ArborScan API", description="Tree analysis server")

# Разрешаем CORS для мобильного клиента
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Пути моделей
BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASSIFIER_MODEL = os.path.join(MODELS_DIR, "classifier.onnx")
STICK_MODEL = os.path.join(MODELS_DIR, "stick_yolo.onnx")
TREE_MODEL = os.path.join(MODELS_DIR, "tree_seg.onnx")

# Загрузка ONNX моделей
print("🔄 Загружаем модели ONNX...")
sess_classifier = ort.InferenceSession(CLASSIFIER_MODEL, providers=["CPUExecutionProvider"])
sess_stick = ort.InferenceSession(STICK_MODEL, providers=["CPUExecutionProvider"])
sess_tree = ort.InferenceSession(TREE_MODEL, providers=["CPUExecutionProvider"])
print("✅ Модели успешно загружены.")


# =========================== ПОЛЕЗНЫЕ ФУНКЦИИ ===========================

def classify_tree(image_rgb: np.ndarray):
    """Определение вида дерева"""
    img_resized = cv2.resize(image_rgb, (224, 224))
    inp = img_resized.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    inputs = {sess_classifier.get_inputs()[0].name: inp}
    preds = sess_classifier.run(None, inputs)[0][0]
    classes = ["Берёза", "Дуб", "Ель", "Сосна", "Тополь"]
    idx = int(np.argmax(preds))
    return classes[idx], float(preds[idx] * 100)


def detect_stick(image_rgb: np.ndarray):
    """Поиск эталонной палки"""
    img_resized = cv2.resize(image_rgb, (640, 640))
    inp = img_resized.astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]

    inputs = {sess_stick.get_inputs()[0].name: inp}
    outputs = sess_stick.run(None, inputs)

    det = outputs[0][0]  # [x1, y1, x2, y2, conf, cls]
    if len(det) < 1:
        return None

    # Берём самую уверенную палку
    det = det[np.argmax(det[:, 4])]
    x1, y1, x2, y2, conf, cls = det
    length_px = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    print(f"📏 Палка найдена, длина = {length_px:.1f}px, conf={conf:.2f}")
    return length_px


def get_weather(lat: float, lon: float, api_key: str):
    """Получаем данные о погоде"""
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric&lang=ru"
        res = requests.get(url, timeout=5).json()
        temp = res["main"]["temp"]
        wind = res["wind"]["speed"]
        gust = res["wind"].get("gust", wind)
        return wind, gust, temp
    except Exception as e:
        print("⚠️ Ошибка получения погоды:", e)
        return None, None, None


# =========================== ГЛАВНАЯ АНАЛИТИКА ===========================

@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float = Form(None),
    lon: float = Form(None),
    api_key: str = Form(""),
):
    try:
        print("📷 Изображение получено")

        # Загружаем изображение
        image_bytes = await file.read()
        img_pil = Image.open(BytesIO(image_bytes)).convert("RGB")
        img = np.array(img_pil)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        h, w = img.shape[:2]

        # --- 1. Определяем вид дерева ---
        species, conf_species = classify_tree(img)
        print(f"🌿 Определён вид: {species} ({conf_species:.1f}% уверенности)")

        # --- 2. Находим эталонную палку ---
        stick_px = detect_stick(img)
        if not stick_px:
            return JSONResponse({"error": "Палка не найдена на фото."})

        stick_length_m = 1.0
        scale = stick_length_m / stick_px  # м/пиксель

        # --- 3. Сегментация дерева ---
        print("🌳 Выполняем сегментацию дерева...")

        img_resized = cv2.resize(img, (640, 640))
        inp = img_resized.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]

        inputs_tree = {sess_tree.get_inputs()[0].name: inp}
        outputs_tree = sess_tree.run(None, inputs_tree)
        mask_data = outputs_tree[1]
        mask = mask_data[0]

        # --- 4. Очистка маски ---
        mask_bin = (mask > 0.35).astype(np.uint8)
        kernel_open = np.ones((5, 5), np.uint8)
        kernel_close = np.ones((7, 7), np.uint8)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel_open)
        mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel_close)

        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return JSONResponse({"error": "Не удалось выделить дерево"})
        cnt = max(contours, key=cv2.contourArea)
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        mask_tree = np.zeros_like(mask_bin)
        cv2.drawContours(mask_tree, [cnt], -1, 1, -1)

        # --- 5. Расчёты ---
        ys, xs = np.where(mask_tree > 0)
        y_bottom = np.max(ys)
        y_top = int(np.percentile(ys, 5))
        tree_height_px = y_bottom - y_top
        tree_height_m = tree_height_px * scale

        # Длина кроны (верхняя треть)
        crown_start = y_top
        crown_end = int(y_top + (tree_height_px / 3))
        crown_m = (crown_end - crown_start) * scale

        # Диаметр ствола (на 5% от низа)
        row_y = int(y_bottom - tree_height_px * 0.05)
        xs_row = np.where(mask_tree[row_y, :] > 0)[0]
        if len(xs_row) > 1:
            dbh_px = xs_row[-1] - xs_row[0]
            dbh_m = dbh_px * scale
        else:
            dbh_m = 0

        # --- 6. Погода (если есть координаты) ---
        if lat and lon and api_key:
            wind, gust, temp = get_weather(lat, lon, api_key)
        else:
            wind, gust, temp = None, None, None

        # --- 7. Визуализация ---
        overlay = img_bgr.copy()
        color_mask = np.zeros_like(img_bgr)
        color_mask[:, :, 1] = (mask_tree * 255)
        vis = cv2.addWeighted(img_bgr, 0.8, color_mask, 0.3, 0)
        cv2.rectangle(vis, (x, y_top), (x + w_box, y_bottom), (255, 0, 0), 2)
        cv2.putText(vis, f"H={tree_height_m:.1f}m", (x+5, y_top+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"D={dbh_m*100:.1f}cm", (x+5, y_bottom-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imwrite("analyzed_tree.png", vis)

        # --- 8. Ответ API ---
        result = {
            "species": species,
            "confidence": conf_species,
            "height_m": round(tree_height_m, 2),
            "crown_m": round(crown_m, 2),
            "dbh_cm": round(dbh_m * 100, 1),
            "wind": wind,
            "gust": gust,
            "temperature": temp,
        }

        print("✅ Анализ завершён успешно.")
        return JSONResponse(result)

    except Exception as e:
        print("❌ Ошибка при анализе:", e)
        return JSONResponse({"error": str(e)})
