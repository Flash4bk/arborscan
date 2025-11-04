# server/stick_detector.py
import cv2
import numpy as np
import onnxruntime as ort

# --------- letterbox как в Ultralytics ----------
def letterbox(img, new_shape=640, color=(114, 114, 114), stride=32, auto=False):
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)

# --------- NMS ----------
def nms(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return []
    boxes = boxes.astype(np.float32)
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        inds = np.where(ovr <= iou_thres)[0]
        order = order[inds + 1]
    return keep

# --------- Декодер вывода YOLOv8-onnx ----------
def decode_yolov8(output, conf_thres=0.15, cls_idx=0):
    """
    Поддержка двух раскладок ONNX:
    - (1, N, 4+num_classes)
    - (1, 4+num_classes, N)
    Возвращает массив [x1,y1,x2,y2], score
    """
    if output.ndim == 3:
        if output.shape[1] > output.shape[2]:
            # (1, 84, N) -> (N, 84)
            pred = output[0].transpose(1, 0)
        else:
            # (1, N, 84) -> (N, 84)
            pred = output[0]
    else:
        raise ValueError(f"Unexpected output shape: {output.shape}")

    boxes = pred[:, :4]
    cls = pred[:, 4:]
    if cls.shape[1] == 1:
        scores = cls[:, 0]
    else:
        # берём только целевой класс 'stick' (index=0)
        scores = cls[:, cls_idx]

    mask = scores >= conf_thres
    boxes = boxes[mask]
    scores = scores[mask]
    return boxes, scores

class StickDetector:
    def __init__(self, onnx_path, img_size=640, providers=None, conf_thres=0.15, iou_thres=0.45):
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        if providers is None:
            providers = ['CPUExecutionProvider']
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.out_name = self.session.get_outputs()[0].name

    def __call__(self, img_bgr):
        h0, w0 = img_bgr.shape[:2]
        img_lb, r, (pad_w, pad_h) = letterbox(img_bgr, self.img_size)

        img = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None]

        output = self.session.run([self.out_name], {self.input_name: img})[0]
        boxes, scores = decode_yolov8(output, conf_thres=self.conf_thres, cls_idx=0)

        if len(boxes) == 0:
            return None, None

        # xywh -> xyxy если нужно
        if boxes.shape[1] == 4 and (boxes[:, 2] <= 1.0).all():
            # иногда экспорт выдаёт относительные координаты, но обычно это xywh на фьюжен-голове.
            # однако в Ultralytics export уже xyxy в пикселях letterbox.
            pass

        # пересчитать координаты из letterbox в исходное изображение
        xyxy = boxes.copy()
        xyxy[:, [0, 2]] -= pad_w
        xyxy[:, [1, 3]] -= pad_h
        xyxy /= r

        # Геометрический фильтр для «рейки»: высокая и узкая, почти вертикальная
        filt_xyxy, filt_scores = [], []
        for b, sc in zip(xyxy, scores):
            x1, y1, x2, y2 = b
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            ar = h / w
            if ar < 3.0:       # должен быть «узкий» столбик
                continue
            if h < 40:         # слишком маленький (на исходном изображении)
                continue
            filt_xyxy.append([x1, y1, x2, y2])
            filt_scores.append(sc)

        if not filt_xyxy:
            return None, None

        filt_xyxy = np.array(filt_xyxy, dtype=np.float32)
        filt_scores = np.array(filt_scores, dtype=np.float32)
        keep = nms(filt_xyxy, filt_scores, iou_thres=self.iou_thres)
        if not keep:
            return None, None

        # берём самый уверенный
        i = keep[0]
        best_box = filt_xyxy[i]
        best_score = float(filt_scores[i])
        return best_box, best_score
