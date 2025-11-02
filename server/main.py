import os
import io
import sys
import cv2
import numpy as np
import onnxruntime as ort
import traceback
from PIL import Image
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

# === Импорты из server ===
from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, get_soil, soil_factor, compute_risk

# === Инициализация FastAPI ===
app = FastAPI(
    title="ArborScan API",
    description="AI-анализ деревьев (вид, параметры, погода, почва, риск)",
    version="1.1"
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

        # === 3. Масштаб по палке ===
        inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]
        res = stick_sess.run(None, {stick_sess.get_inputs()[0].name: inp})
        det = res[0][0]
        if det.shape[0] == 0:
            print("⚠️ Палка не найдена, масштаб принят по умолчанию 1/200.")
            Lpx = 200
        else:
            best = det[np.argmax(det[:, 4])]
            x1, y1, x2, y2 = best[:4]
            Lpx = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * (h0 / 640)
        scale = 1.0 / Lpx
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
        y_top, y_bottom = ys.min(), ys.max()
        H_m = (y_bottom - y_top) * scale

        # === 5. DBH (на высоте груди) ===
        y_dbh = int(y_bottom - 1.3 / scale)
        DBH_m = 0
        if 0 <= y_dbh < mask_bin.shape[0]:
            row = mask_bin[y_dbh, :]
            if np.any(row):
                x_left, x_right = np.where(row > 0)[0][[0, -1]]
                DBH_m = (x_right - x_left) * scale

        # === 6. Реальный диаметр ствола (точное измерение) ===
        bottom_part = mask_bin[int(y_bottom - (y_bottom - y_top) * 0.08):y_bottom, :]
        widths = []
        for y in range(bottom_part.shape[0]):
            row = bottom_part[y, :]
            x_nonzero = np.where(row > 0)[0]
            if len(x_nonzero) > 10:
                widths.append(x_nonzero[-1] - x_nonzero[0])

        if widths:
            avg_width_px = np.median(widths)
            trunk_diameter_m = avg_width_px * scale
        else:
            trunk_diameter_m = DBH_m
        print(f"🪵 Диаметр ствола (у земли): {trunk_diameter_m*100:.1f} см")

        # === 7. Длина кроны ===
        widths_crown = np.array([mask_bin[y, :].sum() for y in range(y_top, y_bottom)], dtype=np.float32)
        dy = np.gradient(widths_crown)
        crown_base_rel = np.argmax(dy > widths_crown.max() * 0.3) if np.any(dy > widths_crown.max() * 0.3) else int(len(widths_crown) * 0.6)
        CL_px = (y_bottom - (y_top + crown_base_rel))
        CL_m = CL_px * scale

        print(f"📏 Высота={H_m:.2f}м, Крона={CL_m:.2f}м, DBH={DBH_m*100:.1f}см")

        # === 8. Погода и почва ===
        wind_speed, gust, temp = get_weather(lat, lon)
        print(f"🌬️ Ветер: {wind_speed} м/с, порывы: {gust} м/с, температура: {temp}°C")
        clay, sand, silt, bd, oc = get_soil(lat, lon)
        k_soil = soil_factor(clay, sand)

        # === 9. Риск ===
        risk, level = compute_risk(species, H_m, DBH_m, CL_m, wind_speed, gust, k_soil)

        # === 10. Визуализация ===
        overlay = img.copy()
        mask_color = np.dstack([np.zeros_like(mask_bin), mask_bin*255, np.zeros_like(mask_bin)]).astype(np.uint8)
        overlay = cv2.addWeighted(overlay, 0.7, mask_color, 0.3, 0)

        # рисуем линию измерения
        y_line = y_bottom - int((y_bottom - y_top) * 0.05)
        cv2.line(overlay, (0, y_line), (w0, y_line), (0, 255, 0), 2)
        text = f"Диаметр: {trunk_diameter_m*100:.1f} см"
        cv2.putText(overlay, text, (30, y_line - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out_path = os.path.join(os.path.dirname(__file__), "analyzed_tree.png")
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Визуализация сохранена: {out_path}")

        # === 11. Ответ ===
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
            "image_path": "analyzed_tree.png"
        }

        sys.stdout.flush()
        print("✅ Анализ завершён успешно.")
        return JSONResponse(content=result, media_type="application/json")

    except Exception as e:
        print("❌ Ошибка при анализе изображения:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)
