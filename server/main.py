import io
import os
import cv2
import json
import time
import numpy as np
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import onnxruntime as ort
from stick_detector import StickDetector


# ------------------- Настройки путей -------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TREE_SEG_MODEL = os.path.join(MODELS_DIR, "tree_seg.onnx")     # сегментация дерева
STICK_MODEL    = os.path.join(MODELS_DIR, "stick_yolo.onnx")   # ваш натренированный детектор палки
VISUAL_NAME    = "analyzed_tree.png"


# ------------------- Приложение -------------------
app = FastAPI(title="ArborScan Server")

# CORS — разрешим всё для простоты
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True
)

# статика для картинок визуализации
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


# ------------------- Загрузка моделей -------------------
print("Загружаем модель ONNX...")
tree_sess = ort.InferenceSession(TREE_SEG_MODEL, providers=["CPUExecutionProvider"])
tree_in = tree_sess.get_inputs()[0]
tree_in_name = tree_in.name
tree_in_shape = tree_in.shape  # [1,3,H,W]
print("Модель успешно загружена.")

stick_detector = StickDetector(STICK_MODEL)


# ------------------- Утилиты -------------------
def read_image_from_upload(file: UploadFile) -> np.ndarray:
    data = file.file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / (np.sum(e) + 1e-9)


def run_tree_segmentation(img_bgr: np.ndarray) -> Optional[np.ndarray]:
    """Возвращает бинарную маску дерева uint8 (0/255) в масштабе исходного изображения."""
    H = int(tree_in_shape[2]); W = int(tree_in_shape[3])

    h0, w0 = img_bgr.shape[:2]
    inp = cv2.resize(img_bgr, (W, H))
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    inp = np.transpose(inp, (2, 0, 1))[None, ...]  # (1,3,H,W)

    out = tree_sess.run(None, {tree_in_name: inp})[0]
    # допустим, out -> (1,1,H,W) logits/score
    if out.ndim == 4 and out.shape[1] == 1:
        prob = 1 / (1 + np.exp(-out[0, 0]))  # sigmoid
    elif out.ndim == 4 and out.shape[1] > 1:
        # много классов -> берём класс "tree" = argmax==1 (пример)
        sm = softmax(out[0], )
        prob = sm[1] if sm.shape[0] > 1 else sm[0]
    else:
        return None

    mask_small = (prob > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask_small, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return mask


def visualize(img_bgr: np.ndarray, tree_mask: Optional[np.ndarray],
              stick: Optional[dict], height_m: Optional[float],
              dbh_cm: Optional[float], out_path: str):
    vis = img_bgr.copy()
    # зелёная маска дерева
    if tree_mask is not None:
        green = np.zeros_like(vis); green[:, :, 1] = 180
        m = (tree_mask > 0)[:, :, None]
        vis = np.where(m, cv2.addWeighted(vis, 0.5, green, 0.5, 0), vis)

    # рамка палки синим
    if stick and "bbox" in stick:
        x1, y1, x2, y2 = stick["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 120, 0), 3)
        cv2.putText(vis, f"stick {stick.get('conf', 0):.2f}",
                    (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 120, 0), 2)

    # подписи
    y = 30
    if height_m is not None:
        cv2.putText(vis, f"H={height_m:.2f} m", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3); y += 36
    if dbh_cm is not None:
        cv2.putText(vis, f"D={dbh_cm:.1f} cm", (14, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)

    cv2.imwrite(out_path, vis)


def estimate_geometry(img_bgr: np.ndarray, tree_mask: Optional[np.ndarray],
                      stick: Optional[dict]) -> tuple[Optional[float], Optional[float]]:
    """
    Очень консервативная оценка:
    - если есть палка -> считаем пиксели/метр и даём физические значения;
    - если палки нет -> вернём None (приложение увидит «нет масштаба»).
    """
    if tree_mask is None:
        return None, None

    ys, xs = np.where(tree_mask > 0)
    if xs.size == 0:
        return None, None

    # высота объекта в пикселях (по bbox маски)
    h_px = (ys.max() - ys.min()) if ys.size > 0 else 0
    if h_px <= 0:
        return None, None

    # масштаб
    meters_per_px = None
    if stick and "bbox" in stick:
        x1, y1, x2, y2 = stick["bbox"]
        stick_h_px = max(1, (y2 - y1))
        # ВАЖНО: здесь поставьте фактическую высоту вашего шеста (метры!)
        STICK_REAL_M = 2.0
        meters_per_px = STICK_REAL_M / stick_h_px

    if meters_per_px is None:
        # нет палки – не считаем физику (пусть будет None)
        return None, None

    height_m = h_px * meters_per_px

    # очень грубая оценка "диаметра" по минимальной ширине маски в нижней трети
    h, w = tree_mask.shape[:2]
    y_lo = int(h * 0.65); y_hi = int(h * 0.95)
    widths = []
    for y in range(y_lo, min(y_hi, h - 1)):
        xs_line = np.where(tree_mask[y] > 0)[0]
        if xs_line.size >= 2:
            widths.append(xs_line.max() - xs_line.min())
    if widths:
        trunk_px = np.percentile(widths, 10)  # нижний перцентиль как «ствол»
        dbh_cm = (trunk_px * meters_per_px) * 100.0
    else:
        dbh_cm = None

    return height_m, dbh_cm


# ------------------- Маршруты -------------------
@app.post("/analyze")
async def analyze(request: Request, file: UploadFile = File(...)):
    t0 = time.time()
    try:
        img = read_image_from_upload(file)
        if img is None:
            return JSONResponse({"ok": False, "error": "bad_image"}, status_code=400)

        # 1) сегментация дерева
        tree_mask = run_tree_segmentation(img)

        # 2) палка (может быть None)
        stick = stick_detector.detect_stick(img)  # -> None или {'bbox':(x1,y1,x2,y2), 'conf':float}

        # 3) оценка геометрии
        height_m, dbh_cm = estimate_geometry(img, tree_mask, stick)

        # 4) визуализация
        out_path = os.path.join(OUTPUT_DIR, VISUAL_NAME)
        visualize(img, tree_mask, stick, height_m, dbh_cm, out_path)

        # 5) формируем ответ
        base = str(request.base_url).rstrip("/")
        vis_url = f"{base}/output/{VISUAL_NAME}"

        # отметки, что данных нет (для вашего UI)
        no_gps = True   # если не передаёте координаты – смело ставим True
        no_soil = True

        result = {
            "ok": True,
            "species": {"name": "Берёза", "confidence": 0.369},  # заглушка; интегрируйте свой классификатор
            "metrics": {
                "height_m": round(height_m, 3) if height_m is not None else None,
                "dbh_cm": round(dbh_cm, 1) if dbh_cm is not None else None,
                "crown_m": None
            },
            "scale": {
                "stick_found": bool(stick),
                "note": None if stick else "Масштаб неизвестен: палка не найдена"
            },
            "weather": {
                "wind": None, "gust": None, "temp": None,
                "note": "Нет данных GPS, пропущен анализ погоды и почвы" if no_gps else None
            },
            "risk": {"level": "Низкий", "score": 0.0},
            "visualization_url": vis_url,
            "time_ms": int((time.time() - t0) * 1000)
        }
        return JSONResponse(result)

    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/health")
def health():
    return {"ok": True}
