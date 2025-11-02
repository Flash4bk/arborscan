import os
import io
import sys
import cv2
import json
import numpy as np
import onnxruntime as ort
import traceback
from PIL import Image, ExifTags
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse

# Импорт внутренних модулей
from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, get_soil, soil_factor, compute_risk

app = FastAPI(
    title="ArborScan API",
    description="Вид, геометрия, погода, почва, риск и визуализация дерева",
    version="2.4"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TREE_MODEL = os.path.join(MODEL_DIR, "tree_seg.onnx")
STICK_MODEL = os.path.join(MODEL_DIR, "stick_yolo.onnx")

# ------------------------------
#  Вспомогательные функции
# ------------------------------

def _to_deg(value):
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    return d + (m / 60.0) + (s / 3600.0)

def extract_gps_from_exif(pil_img: Image.Image):
    try:
        exif = pil_img._getexif()
        if not exif:
            return None, None
        gps_tag = [k for k, v in ExifTags.TAGS.items() if v == "GPSInfo"]
        if not gps_tag:
            return None, None
        gps_data = exif.get(gps_tag[0])
        if not gps_data:
            return None, None

        lat_ref = gps_data.get(1)
        lat_val = gps_data.get(2)
        lon_ref = gps_data.get(3)
        lon_val = gps_data.get(4)
        if lat_ref and lat_val and lon_ref and lon_val:
            lat = _to_deg(lat_val)
            if lat_ref in ["S", "s"]:
                lat = -lat
            lon = _to_deg(lon_val)
            if lon_ref in ["W", "w"]:
                lon = -lon
            return lat, lon
    except Exception:
        pass
    return None, None

def postprocess_mask(mask_bin: np.ndarray) -> np.ndarray:
    """Очистка маски дерева"""
    mask_bin = (mask_bin > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return mask_bin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, kernel, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)

# ------------------------------
#  Загрузка моделей
# ------------------------------
try:
    print("🔄 Загружаем модели ONNX...")
    tree_seg_sess = ort.InferenceSession(TREE_MODEL, providers=["CPUExecutionProvider"])
    stick_sess = ort.InferenceSession(STICK_MODEL, providers=["CPUExecutionProvider"])
    print("✅ Модели успешно загружены.")
except Exception as e:
    print("❌ Ошибка загрузки моделей:", e)

# ------------------------------
#  Основные эндпоинты
# ------------------------------

@app.get("/")
def root():
    return {"message": "🌲 ArborScan API активен. Используйте POST /analyze."}


@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None)
):
    """Основной эндпоинт анализа дерева"""
    try:
        # ---------- 1. Загрузка изображения ----------
        image_bytes = await file.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = np.array(pil)
        w0, h0 = pil.size
        print(f"📷 Изображение получено: {w0}x{h0}")

        # GPS из EXIF при отсутствии координат
        if lat is None or lon is None:
            exif_lat, exif_lon = extract_gps_from_exif(pil)
            lat = lat if lat is not None else exif_lat
            lon = lon if lon is not None else exif_lon

        # ---------- 2. Определение вида ----------
        temp_path = os.path.join(os.path.dirname(__file__), "temp.jpg")
        with open(temp_path, "wb") as f:
            f.write(image_bytes)
        species, conf = classify_tree(temp_path)
        print(f"🌿 Определён вид: {species} ({conf*100:.1f}% уверенности)")

        # ---------- 3. Поиск палки ----------
        inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]
        res = stick_sess.run(None, {stick_sess.get_inputs()[0].name: inp})
        det = res[0][0]
        stick_box_640 = None
        scale = None

        if det.shape[0] > 0:
            best_idx = None
            best_score = -1.0
            for i in range(det.shape[0]):
                x1, y1, x2, y2, s = det[i][:5]  # берём только первые 5 значений
                w = max(x2 - x1, 1e-3)
                h = max(y2 - y1, 1e-3)
                vert = h / w
                score = float(s) * vert * h
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                x1, y1, x2, y2, s = det[best_idx][:5]
                stick_box_640 = (float(x1), float(y1), float(x2), float(y2))
                sx, sy = w0 / 640.0, h0 / 640.0
                x1o, y1o, x2o, y2o = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                stick_h_px = max(y2o - y1o, 1)
                scale = 1.0 / stick_h_px  # м/пикс
                print(f"📏 Палка найдена: {stick_h_px}px → {scale:.5f} м/пикс")
        else:
            print("⚠️ Палка не найдена, используется масштаб 0.003 м/пикс (по умолчанию)")
            scale = 0.003

        if not (0 < scale < 0.02):
            scale = 0.003

        # ---------- 4. Сегментация дерева ----------
        seg_inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        seg_inp = np.transpose(seg_inp, (2, 0, 1))[None, :, :, :]
        seg_out = tree_seg_sess.run(None, {tree_seg_sess.get_inputs()[0].name: seg_inp})
        mask = seg_out[0][0].mean(axis=0)
        if mask is None:
            raise RuntimeError("Сегментация не вернула маску.")

        mask = cv2.resize(mask, (w0, h0))
        mask_bin = (mask > np.percentile(mask, 85)).astype(np.uint8)
        mask_bin = postprocess_mask(mask_bin)

        ys, xs = np.where(mask_bin > 0)
        if len(ys) == 0:
            raise RuntimeError("Не удалось выделить дерево на фото.")
        x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
        h = y_max - y_min
        H_m = h * scale

        # Диаметр у земли
        y_bottom = y_max
        y_start = int(y_max - h * 0.1)
        region = mask_bin[y_start:y_bottom, x_min:x_max]
        proj = np.sum(region, axis=0)
        nz = np.where(proj > 0)[0]
        trunk_diam_m = (nz[-1] - nz[0]) * scale if len(nz) > 1 else 0.0

        # DBH
        y_dbh = y_bottom - int(1.3 / scale)
        if 0 <= y_dbh < h0:
            row = mask_bin[y_dbh, x_min:x_max]
            nz2 = np.where(row > 0)[0]
            DBH_m = ((nz2[-1] - nz2[0]) * scale) if len(nz2) > 1 else trunk_diam_m
        else:
            DBH_m = trunk_diam_m

        # Крона
        widths = np.array([mask_bin[r, :].sum() for r in range(y_min, y_max)], dtype=np.float32)
        if widths.size > 0:
            max_w = float(np.max(widths))
            thr = max_w * 0.6
            idx = np.argmax(widths < thr)
            CL_px = (h - idx) if idx > 0 else int(h * 0.6)
        else:
            CL_px = int(h * 0.6)
        CL_m = CL_px * scale

        print(f"📐 H={H_m:.2f}м, D={trunk_diam_m*100:.1f}см, DBH={DBH_m*100:.1f}см, Крона={CL_m:.2f}м")

        # ---------- 5. Погода и почва ----------
        weather_json = {"message": "Нет данных: фото без GPS."}
        soil_json = {"message": "Нет данных: фото без GPS."}
        k_soil = 1.0

        if lat is not None and lon is not None:
            try:
                wind, gust, temp = get_weather(lat, lon)
                weather_json = {"wind": wind, "gust": gust, "temp": temp}
            except Exception as e:
                weather_json = {"message": f"Нет данных (погода недоступна): {e.__class__.__name__}"}

            try:
                clay, sand, silt, bd, oc = get_soil(lat, lon)
                k_soil = soil_factor(clay, sand)
                soil_json = {"clay": clay, "sand": sand, "k_soil": k_soil}
            except Exception as e:
                soil_json = {"message": f"Нет данных (почва недоступна): {e.__class__.__name__}"}

        # ---------- 6. Риск ----------
        wind_val = weather_json.get("wind", 0.0) if isinstance(weather_json, dict) else 0.0
        gust_val = weather_json.get("gust", 0.0) if isinstance(weather_json, dict) else 0.0
        risk, level = compute_risk(species, H_m, DBH_m, CL_m, wind_val, gust_val, k_soil)

        # ---------- 7. Визуализация ----------
        vis = img.copy()
        mask_color = np.dstack([
            np.zeros_like(mask_bin),
            (mask_bin * 255).astype(np.uint8),
            np.zeros_like(mask_bin)
        ])
        vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        cv2.rectangle(vis, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
        cv2.putText(vis, f"H={H_m:.1f}m", (x_min, max(15, y_min - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"D={trunk_diam_m*100:.1f}cm", (x_min, y_bottom + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if stick_box_640 is not None:
            sx, sy = w0 / 640.0, h0 / 640.0
            x1, y1, x2, y2 = stick_box_640
            x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(vis, "Палка (1м)", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        out_path = os.path.join(os.path.dirname(__file__), "analyzed_tree.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Визуализация сохранена: {out_path}")

        # ---------- 8. Ответ ----------
        return JSONResponse({
            "species": species,
            "confidence": round(conf * 100, 1),
            "height_m": round(H_m, 2),
            "crown_len_m": round(CL_m, 2),
            "dbh_cm": round(DBH_m * 100, 1),
            "trunk_diameter_cm": round(trunk_diam_m * 100, 1),
            "weather": weather_json,
            "soil": soil_json,
            "risk": {"score": round(risk, 1), "level": level},
            "gps": {"lat": lat, "lon": lon},
            "image_path": "/image"
        })

    except Exception as e:
        print("❌ Ошибка при анализе:")
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/image")
def get_result_image():
    out_path = os.path.join(os.path.dirname(__file__), "analyzed_tree.png")
    if not os.path.exists(out_path):
        return JSONResponse({"error": "Визуализация отсутствует"}, status_code=404)
    return FileResponse(out_path, media_type="image/png", filename="analyzed_tree.png")
