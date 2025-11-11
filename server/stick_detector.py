import cv2
import numpy as np
import onnxruntime as ort
import logging

logger = logging.getLogger(__name__)


def _to_float(x) -> float:
    """Аккуратно вытащить float из чего угодно (numpy, массив и т.п.)."""
    if isinstance(x, np.ndarray):
        if x.size == 0:
            return 0.0
        x = x.flatten()[0]
    try:
        return float(x)
    except Exception:
        return 0.0


class StickDetector:
    """
    Универсальная обёртка ONNX-модели для детекта «палка» (калибровочный шест).
    Возвращает либо None, либо dict: {"bbox":(x1,y1,x2,y2), "conf":float}.
    Никаких распаковок кортежей – только словарь.
    """

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0]
        self.input_name = self.input.name
        self.input_shape = self.input.shape  # [1,3,H,W] чаще всего
        logger.info(f"StickDetector загружен: {self.input_shape}")

    def _preprocess(self, img_bgr: np.ndarray) -> np.ndarray:
        # ожидаем [1,3,H,W]
        H = int(self.input_shape[2])
        W = int(self.input_shape[3])
        img = cv2.resize(img_bgr, (W, H))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)
        img = np.expand_dims(img, 0)        # (1,3,H,W)
        return img

    def detect_stick(self, img_bgr: np.ndarray):
        """
        Пытаемся получить верхний по уверенности бокс палки.
        Поддерживает разные форматы выхода: [N,4..6], [1,N,6], и т.п.
        Если ничего нет – возвращает None.
        """
        try:
            inp = self._preprocess(img_bgr)
            outs = self.session.run(None, {self.input_name: inp})
            if not outs or outs[0] is None:
                logger.warning("StickDetector: пустой выход модели.")
                return None

            det = outs[0]
            det = np.array(det)

            # Нормализуем форму к [N, M]
            if det.ndim == 3:
                # например [1, N, 6]
                det = det.reshape(-1, det.shape[-1])
            elif det.ndim == 1:
                # один детект
                det = det.reshape(1, -1)

            if det.size == 0 or det.shape[0] == 0:
                logger.warning("StickDetector: детекций нет.")
                return None

            # Ожидаем минимум 5 значений: x1,y1,x2,y2,conf  (+ возможно class)
            if det.shape[1] < 5:
                logger.warning(f"StickDetector: неожиданная форма выхода {det.shape}")
                return None

            # Берём детект с максимальной уверенность (столбец 4 чаще всего conf)
            conf_col = 4 if det.shape[1] >= 5 else det.shape[1] - 1
            best_idx = int(np.argmax(det[:, conf_col]))
            best = det[best_idx]

            x1 = _to_float(best[0])
            y1 = _to_float(best[1])
            x2 = _to_float(best[2])
            y2 = _to_float(best[3])
            conf = _to_float(best[conf_col])

            # sanity check
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                return None

            return {"bbox": (int(x1), int(y1), int(x2), int(y2)), "conf": conf}

        except Exception as e:
            logger.error(f"Ошибка StickDetector: {e}")
            return None
