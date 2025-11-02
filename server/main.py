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

# наши вспомогательные модули
from server.classify_tree import classify_tree
from server.risk_analysis import get_weather, get_soil, soil_factor, compute_risk

app = FastAPI(
    title="ArborScan API",
    description="Вид, геометрия, погода/почвы, риск + визуализация",
    version="2.3"
)

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
TREE_MODEL = os.path.join(MODEL_DIR, "tree_seg.onnx")
STICK_MODEL = os.path.join(MODEL_DIR, "stick_yolo.onnx")

# ------------------------------
#  Utility: EXIF -> GPS
# ------------------------------
def _to_deg(value):
    # value: ((deg_num,deg_den),(min_num,min_den),(sec_num,sec_den))
    d = value[0][0] / value[0][1]
    m = value[1][0] / value[1][1]
    s = value[2][0] / value[2][1]
    return d + m / 60.0 + s / 3600.0

def extract_gps_from_exif(pil_img: Image.Image):
    try:
        exif = pil_img._getexif()
        if not exif:
            return None, None
        gps_tag = None
        for k, v in ExifTags.TAGS.items():
            if v == 'GPSInfo':
                gps_tag = k
                break
        gps = exif.get(gps_tag)
        if not gps:
            return None, None

        lat = lon = None
        lat_ref = gps.get(1)
        lat_val = gps.get(2)
        lon_ref = gps.get(3)
        lon_val = gps.get(4)
        if lat_ref and lat_val and lon_ref and lon_val:
            lat = _to_deg(lat_val)
            if lat_ref in ['S', 's']: lat = -lat
            lon = _to_deg(lon_val)
            if lon_ref in ['W', 'w']: lon = -lon
        return lat, lon
    except:
        return None, None

# ------------------------------
#  Postprocess tree mask
# ------------------------------
def postprocess_mask(mask_bin: np.ndarray) -> np.ndarray:
    """morph close/open + largest connected component"""
    mask_bin = (mask_bin > 0).astype(np.uint8)
    if mask_bin.sum() == 0:
        return mask_bin
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(mask_bin, cv2.MORPH_CLOSE, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return m
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (labels == largest).astype(np.uint8)

# ------------------------------
#  Load sessions
# ------------------------------
try:
    print("🔄 Загружаем модели ONNX...")
    tree_seg_sess = ort.InferenceSession(TREE_MODEL, providers=["CPUExecutionProvider"])
    stick_sess    = ort.InferenceSession(STICK_MODEL, providers=["CPUExecutionProvider"])
    print("✅ Модели успешно загружены.")
except Exception as e:
    print("❌ Ошибка загрузки моделей:", e)


@app.get("/")
def root():
    return {"message": "🌲 ArborScan API активен. POST /analyze — анализ фото."}


@app.post("/analyze")
async def analyze_tree(
    file: UploadFile = File(...),
    lat: float | None = Form(default=None),
    lon: float | None = Form(default=None)
):
    """
    Если lat/lon не пришли — пытаемся достать из EXIF.
    Если координат нет — возвращаем weather/soil: "Нет данных...".
    """
    try:
        # ------------ 1) load image ------------
        image_bytes = await file.read()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w0, h0 = pil.size
        img = np.array(pil)  # RGB
        print(f"📷 Изображение получено: {w0}x{h0}")

        # GPS fallback из EXIF
        if lat is None or lon is None:
            exif_lat, exif_lon = extract_gps_from_exif(pil)
            lat = lat if lat is not None else exif_lat
            lon = lon if lon is not None else exif_lon

        # ------------ 2) classify ------------
        with open(os.path.join(os.path.dirname(__file__), "temp.jpg"), "wb") as f:
            f.write(image_bytes)
        species, conf = classify_tree(os.path.join(os.path.dirname(__file__), "temp.jpg"))
        print(f"🌿 Вид: {species} ({conf*100:.1f}%)")

        # ------------ 3) detect stick for scale ------------
        
        
            # resize to 640x640 (без letterbox — как и раньше)
        inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, :, :, :]
        res = stick_sess.run(None, {stick_sess.get_inputs()[0].name: inp})

                    # YOLO может вернуть разное количество параметров: [x1,y1,x2,y2,conf,(cls)...]
        det = res[0][0]
        stick_box_640 = None
        scale = None

        if det.shape[0] > 0:
                # берём самую "вертикальную" и высокую рамку
                best_idx = None
                best_score = -1.0
                for i in range(det.shape[0]):
                    # безопасно обрезаем до первых 5 элементов
                    x1, y1, x2, y2, s = det[i][:5]
                    w = max(x2 - x1, 1e-3)
                    h = max(y2 - y1, 1e-3)
                    vert = h / w
                    score = float(s) * vert * h
                    if score > best_score:
                        best_score = score
                        best_idx = i

                # если найден лучший детект
                if best_idx is not None:
                    x1, y1, x2, y2, s = det[best_idx][:5]
                    stick_box_640 = (float(x1), float(y1), float(x2), float(y2))

                    # масштабирование из 640x640 → оригинальное изображение
                    sx, sy = w0 / 640.0, h0 / 640.0
                    x1o, y1o, x2o, y2o = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
                    stick_h_px = max(y2o - y1o, 1)
                    # 1 метр реальной длины = высота рамки
                    scale = 1.0 / stick_h_px  # м/пикс
                    print(f"📏 Палка: {stick_h_px}px по высоте → масштаб {scale:.5f} м/пикс")
        else:
                print("⚠️ Палка не найдена, ставлю усреднённый масштаб 0.003 м/пикс (≈3 мм)")
                scale = 0.003

        if not (0 < scale < 0.02):  # предохранитель от ошибок масштаба
                scale = 0.003



        # ------------ 4) tree segmentation ------------
        seg_inp = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        seg_inp = np.transpose(seg_inp, (2, 0, 1))[None, :, :, :]
        seg_out = tree_seg_sess.run(None, {tree_seg_sess.get_inputs()[0].name: seg_inp})
        protos = seg_out[1][0] if len(seg_out) > 1 else None
        if protos is None:
            raise RuntimeError("Сегментация не вернула маску.")

        mask = protos.mean(axis=0)
        mask = cv2.resize(mask, (w0, h0))
        mask_bin = (mask > np.percentile(mask, 85)).astype(np.uint8)

        # постпроцесс: морфология + largest component
        mask_bin = postprocess_mask(mask_bin)

        # если ещё шум — ограничимся окном вокруг основной компоненты
        ys, xs = np.where(mask_bin > 0)
        if len(ys) == 0:
            raise RuntimeError("Не удалось выделить дерево на фото.")
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()
        # дополнительно расширим окно немного
        pad_y = int(0.05 * (y_max - y_min + 1))
        pad_x = int(0.05 * (x_max - x_min + 1))
        y0c, y1c = max(0, y_min - pad_y), min(h0, y_max + pad_y + 1)
        x0c, x1c = max(0, x_min - pad_x), min(w0, x_max + pad_x + 1)
        # на всякий случай обнулим всё вне окна
        crop_mask = np.zeros_like(mask_bin)
        crop_mask[y0c:y1c, x0c:x1c] = mask_bin[y0c:y1c, x0c:x1c]
        mask_bin = crop_mask

        # контур и bbox
        contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        tree_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(tree_contour)

        # ------------ 5) geometry ------------
        H_m = h * scale

        # диаметр у земли — по нижним 10% bbox
        y_bottom = y + h
        y_start = y + int(0.9 * h)
        y_start = min(max(y_start, 0), h0 - 1)
        region = mask_bin[y_start:y_bottom, x:x+w]
        proj = np.sum(region, axis=0)
        nz = np.where(proj > 0)[0]
        if len(nz) > 1:
            trunk_px = nz[-1] - nz[0]
            trunk_diameter_m = trunk_px * scale
        else:
            trunk_diameter_m = 0.0

        # DBH: 1.3 м от земли
        y_dbh = y_bottom - int(1.3 / scale)
        if 0 <= y_dbh < h0:
            row = mask_bin[y_dbh, x:x+w]
            nzd = np.where(row > 0)[0]
            DBH_m = ((nzd[-1] - nzd[0]) * scale) if len(nzd) > 1 else trunk_diameter_m
        else:
            DBH_m = trunk_diameter_m

        # длина кроны: где поперечная ширина падает на 40% и более от max
        vert_widths = np.array([mask_bin[r, :].sum() for r in range(y, y + h)], dtype=np.float32)
        if vert_widths.size > 0:
            max_w = float(np.max(vert_widths))
            thr = max_w * 0.6
            idx = np.argmax(vert_widths < thr)
            CL_px = (h - idx) if idx > 0 else int(h * 0.6)
        else:
            CL_px = int(h * 0.6)
        CL_m = CL_px * scale

        print(f"📐 H={H_m:.2f}м, Crown={CL_m:.2f}м, D_ground={trunk_diameter_m*100:.1f}см, DBH={DBH_m*100:.1f}см")

        # ------------ 6) weather / soil with graceful fallback ------------
        weather_json = {"message": "Нет данных: фото без GPS."}
        soil_json    = {"message": "Нет данных: фото без GPS."}
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

        # ------------ 7) risk ------------
        wind_val = weather_json.get("wind", 0.0) if isinstance(weather_json, dict) else 0.0
        gust_val = weather_json.get("gust", 0.0) if isinstance(weather_json, dict) else 0.0
        risk, level = compute_risk(species, H_m, DBH_m, CL_m, wind_val, gust_val, k_soil)

        # ------------ 8) visualization ------------
        vis = img.copy()

        # зелёная полупрозрачная маска
        mask_color = np.dstack([
            np.zeros_like(mask_bin),
            (mask_bin * 255).astype(np.uint8),
            np.zeros_like(mask_bin)
        ])
        vis = cv2.addWeighted(vis, 0.7, mask_color, 0.3, 0)

        # контур и bbox
        cv2.drawContours(vis, [tree_contour], -1, (0, 255, 0), 2)
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # линия DBH
        y_dbh_line = int(np.clip(y_bottom - int(1.3 / scale), 0, h0 - 1))
        cv2.line(vis, (x, y_dbh_line), (x + w, y_dbh_line), (0, 255, 255), 2)

        # палка (если была)
        if stick_box_640 is not None:
            sx, sy = w0 / 640.0, h0 / 640.0
            x1, y1, x2, y2 = stick_box_640
            x1, y1, x2, y2 = int(x1 * sx), int(y1 * sy), int(x2 * sx), int(y2 * sy)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cv2.putText(vis, "Палка (1м)", (x1, max(10, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(vis, f"H={H_m:.1f}m", (x, max(15, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(vis, f"D={trunk_diameter_m*100:.1f}cm", (x, min(h0 - 10, y_bottom + 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = os.path.join(os.path.dirname(__file__), "analyzed_tree.png")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        print(f"🖼️ Визуализация сохранена: {out_path}")

        # ------------ 9) response ------------
        result = {
            "species": species,
            "confidence": round(conf * 100, 1),
            "height_m": round(H_m, 2),
            "crown_len_m": round(CL_m, 2),
            "dbh_cm": round(DBH_m * 100, 1),
            "trunk_diameter_cm": round(trunk_diameter_m * 100, 1),
            "weather": weather_json,
            "soil": soil_json,
            "risk": {"score": round(risk, 1), "level": level},
            "image_path": "/image",
            "gps": {"lat": lat, "lon": lon}
        }

        print("✅ Анализ завершён.")
        return JSONResponse(result)

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
