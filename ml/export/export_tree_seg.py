import os
from ultralytics import YOLO
import shutil, glob

BASE_DIR = os.path.dirname(__file__)
SRC_PATH = os.path.join(BASE_DIR, "..", "models_src", "tree_model.pt")
OUT_DIR = os.path.join(BASE_DIR, "..", "onnx")

# === Экспорт YOLOv8-seg ===
print("🚀 Exporting tree_model.pt to ONNX...")
model = YOLO(SRC_PATH)
model.export(format="onnx", imgsz=640, opset=17, dynamic=True)

# === Перемещаем результат в нужную папку ===
os.makedirs(OUT_DIR, exist_ok=True)
onnx_path = glob.glob("*.onnx")
for f in onnx_path:
    if "tree" in f:
        shutil.move(f, os.path.join(OUT_DIR, "tree_seg.onnx"))
print(f"✅ Saved: {OUT_DIR}\\tree_seg.onnx")
