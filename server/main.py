import os
import io
import sys
import cv2
import numpy as np
import onnxruntime as ort
import traceback
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse

# === Импорты из server ===
from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, get_soil, soil_factor, compute_risk

# === Инициализация FastAPI ===
app = FastAPI(
    title="ArborScan API",
    description="AI-анализ деревьев (вид, параметры, погода, почва, риск) + визуализация",
    version="2.0"
)

# === Пути к моделям ===
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TREE_MODEL = os.path.join(MODEL_DIR, "tree_seg.onnx")
STICK_MODEL = os.path.join(MODEL_DIR, "stick_yolo.onnx")

# === Загрузка моделей ===
try:
    print("🔄 Загружаем модели ONNX...")
    tree_seg_sess = ort.InferenceSession(TREE_MODEL, providers=["CPUExecutionProvider"])
    stick_sess = ort.InferenceSession(STICK_MODEL, providers=["CPUExecutionProvider"])
    print("✅ Модели успешно загружены.")
except Exception as e:
    print("❌ Ошибка загрузки моделей:", e)


@app.get("/")
def root():
    return {"message": "🌲 ArborScan API работает! Используйте POST /analyze для анализа."}


@app.post("/analyze")
async def analyze_tree(file: UploadFile = File(...), lat: float = 55.75, lon: float = 37.62):
    try:
        # === 1. Загружаем изображение ===
        image_bytes = await file.read()
        img = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        h0, w0 = img.shape[:2]
        print(f"📷 Изображение получено: {w0}x{h0}")

        # === 2. Классификация дерева ===
        with open("temp.jpg", "wb") as f:
            f.write(image_bytes)
        species, conf = classify_tree("temp.jpg")
        print(f"🌿 Определён вид: {species} ({conf*100:.1f}% уверенности)")

        # === 3. Масштаб по палке (YOLOv8) ===
        inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]
        res = stick_sess.run(None, {stick_sess.get_inputs()[0].name: inp})
        det = res[0][0]

        if det.shape[0] == 0:
            print("⚠️ Палка не найдена, используем усреднённый масштаб 0.003 м/пиксель (~3 мм)")
            scale = 0.003
        else:
            best = det[np.argmax(det[:, 4])]
            x1, y1, x2, y2 = best[:4]
            stick_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            scale = 1.0 / max(stick_px, 1)
            print(f"📏 Эталонная палка: {stick_px:.1f}px, масштаб: {scale:.5f} м/пиксель")

        if scale <= 0 or scale > 0.02:
            print("⚠️ Масштаб недостоверен, заменён на усреднённый (0.003 м/пиксель)")
            scale = 0.003

        # === 4. Сегментация дерева ===
        tree_inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        tree_inp = np.transpose(tree_inp, (2, 0, 1))[None, :, :, :]
        res = tree_seg_sess.run(None, {tree_seg_sess.get_inputs()[0].name: tree_inp})
        protos = res[1][0] if len(res) > 1 else None
        if protos is None:
            raise RuntimeError("Модель сегментации не вернула маску.")
        mask = protos.mean(axis=0)
        mask = cv2.resize(mask, (w0, h0))
        mask_bin = (mask > np.percentile(mask, 85)).astype(np.uint8)

        ys, xs = np.where(mask_bin > 0)
        if len(ys) == 0:
            raise RuntimeError("Не удалось выделить дерево на фото.")

        # === 5. Геометрия дерева ===
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise RuntimeError("Контур дерева не найден.")
        tree_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(tree_contour)

        # --- высота ---
        H_m = h * scale
        print(f"📐 Высота дерева: {H_m:.2f} м")

        # --- диаметр у земли ---
        y_start = int(y + h * 0.9)
        y_end = y + h
        region = mask_bin[y_start:y_end, x:x+w]
        proj = np.sum(region, axis=0)
        nonzero = np.where(proj > 0)[0]
        if len(nonzero) > 1:
            trunk_width_px = nonzero[-1] - nonzero[0]
            trunk_diameter_m = trunk_width_px * scale
        else:
            trunk_diameter_m = 0
        print(f"🪵 Диаметр ствола (у земли): {trunk_diameter_m*100:.1f} см")

        # --- DBH (1.3 м от земли) ---
        y_dbh = int(y_end - 1.3 / scale)
        if 0 <= y_dbh < mask_bin.shape[0]:
            dbh_row = mask_bin[y_dbh, x:x+w]
            nz = np.where(dbh_row > 0)[0]
            if len(nz) > 1:
                DBH_m = (nz[-1] - nz[0]) * scale
            else:
                DBH_m = trunk_diameter_m
        else:
            DBH_m = trunk_diameter_m
        print(f"🪵 DBH (1.3м): {DBH_m*100:.1f} см")

        # --- длина кроны ---
        vertical_widths = np.array([mask_bin[y0, :].sum() for y0 in range(y, y+h)], dtype=np.float32)
        max_w = np.max(vertical_widths)
        threshold = max_w * 0.6
        crown_top_idx = np.argmax(vertical_widths < threshold)
        CL_px = (h - crown_top_idx) if crown_top_idx > 0 else int(h * 0.6)
        CL_m = CL_px * scale
        print(f"🌿 Длина кроны: {CL_m:.2f} м")

        # === 6. Погода и почва ===
        wind_speed, gust, temp = get_weather(lat, lon)
        print(f"🌬️ Ветер: {wind_speed} м/с, порывы: {gust} м/с, температура: {temp}°C")
        clay, sand, silt, bd, oc = get_soil(lat, lon)
        k_soil = soil_factor(clay, sand)

        # === 7. Риск ===
        risk, level = compute_risk(species, H_m, DBH_m, CL_m, wind_speed, gust, k_soil)

        # === 8. Визуализация ===
        vis = img.copy()
        cv2.drawContours(vis, [tree_contour], -1, (0, 255, 0), 2)
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.line(vis, (x, y_end - int(1.3 / scale)), (x+w, y_end - int(1.3 / scale)), (0, 255, 255), 2)
        cv2.putText(vis, f"H={H_m:.1f}m", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"D={trunk_diameter_m*100:.1f}cm", (x, y_end+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out_path = os.path.join(os.path.dirname(__file__), "analyzed_tree.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Визуализация сохранена: {out_path}")

        # === 9. Результат ===
        result = {
            "species": species,
            "confidence": round(conf * 100, 1),
            "height_m": round(H_m, 2),
            "crown_len_m": round(CL_m, 2),
            "dbh_cm": round(DBH_m * 100, 1),
            "trunk_diameter_cm": round(trunk_diameter_m * 100, 1),
            "weather": {"wind": wind_speed, "gust": gust, "temp": temp},
            "soil": {"clay": clay, "sand": sand, "k_soil": k_soil},
            "risk": {"score": round(risk, 1), "level": level},
            "image_path": "/image"
        }

        sys.stdout.flush()
        print("✅ Анализ завершён успешно.")
        return JSONResponse(content=result, media_type="application/json")

    except Exception as e:
        print("❌ Ошибка при анализе изображения:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/image")
def get_result_image():
    """Позволяет скачать последнюю визуализацию"""
    out_path = os.path.join(os.path.dirname(__file__), "analyzed_tree.png")
    if not os.path.exists(out_path):
        return JSONResponse({"error": "Нет визуализации"}, status_code=404)
    return FileResponse(out_path, media_type="image/png", filename="analyzed_tree.png")
