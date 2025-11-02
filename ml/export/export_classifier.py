import os
import torch
import torch.nn as nn
from torchvision.models import resnet18

# === Путь до текущей папки (ml/export) ===
BASE_DIR = os.path.dirname(__file__)

# === Пути к моделям и выходам ===
SRC_PATH = os.path.join(BASE_DIR, "..", "models_src", "classifier.pth")
OUT_PATH = os.path.join(BASE_DIR, "..", "onnx", "classifier.onnx")

# === Количество классов ===
num_classes = 5  # [Берёза, Дуб, Ель, Сосна, Тополь]

# === Загружаем ResNet18 и веса ===
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
state = torch.load(SRC_PATH, map_location="cpu")
model.load_state_dict(state)
model.eval()

# === Экспорт ===
dummy = torch.randn(1, 3, 224, 224)
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

torch.onnx.export(
    model,
    dummy,
    OUT_PATH,
    input_names=["image"],
    output_names=["probs"],
    opset_version=17,
    dynamic_axes={"image": {0: "N"}, "probs": {0: "N"}}
)
print(f"✅ Saved: {OUT_PATH}")
