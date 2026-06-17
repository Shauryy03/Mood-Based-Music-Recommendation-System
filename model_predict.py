# model_predict.py  -- improved face cropping, optional alignment, safe softmax, CLAHE preprocessing
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import math

# -------------------------------
# Emotion labels (keep same mapping as your model training)
# -------------------------------
EMOTION_LABELS = {
    0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
    4: 'sad', 5: 'surprise', 6: 'neutral'
}

# -------------------------------
# MediaPipe detectors (reusable)
# -------------------------------
_mp_face_det = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.6)
_mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                               refine_landmarks=True, min_detection_confidence=0.6)

# -------------------------------
# Load FER model
# Returns model, input_shape (H,W,C) and whether last activation is softmax
# -------------------------------
def load_fer_model(path='fer_model_final.h5'):
    model = tf.keras.models.load_model(path, compile=False)
    input_shape = model.input_shape[1:]  # (H, W, C)
    # try to detect if final activation is softmax
    last = model.layers[-1]
    has_softmax = False
    try:
        act = getattr(last, 'activation', None)
        if act is not None and getattr(act, '__name__', '') == 'softmax':
            has_softmax = True
    except Exception:
        has_softmax = False
    return model, input_shape, has_softmax

# -------------------------------
# utility: safe softmax (numerically stable)
# -------------------------------
def softmax(x):
    x = np.array(x, dtype=np.float64)
    e = np.exp(x - np.max(x))
    return e / e.sum()

# -------------------------------
# Face crop + optional alignment
# - Uses MediaPipe face detection for bbox
# - Returns a square crop (preserves aspect), applies padding
# - Optionally aligns using face-mesh eye landmarks
# -------------------------------
def detect_and_crop_face(frame, use_mediapipe=True, pad_ratio=0.25, align=True, min_face_size=40):
    h_frame, w_frame = frame.shape[:2]
    face_crop = None

    if use_mediapipe:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = _mp_face_det.process(rgb)
        if results.detections:
            det = results.detections[0]
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w_frame)
            y = int(bbox.ymin * h_frame)
            w = int(bbox.width * w_frame)
            h = int(bbox.height * h_frame)

            # size guard
            if w < min_face_size or h < min_face_size:
                return None

            # square crop around face center
            side = max(w, h)
            pad = int(side * pad_ratio)
            cx = x + w // 2
            cy = y + h // 2
            half = side // 2
            x1 = max(0, cx - half - pad)
            y1 = max(0, cy - half - pad)
            x2 = min(w_frame, cx + half + pad)
            y2 = min(h_frame, cy + half + pad)

            face_crop = frame[y1:y2, x1:x2].copy()

            # alignment using face_mesh on the crop
            if align and face_crop.size != 0:
                try:
                    crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    mesh_res = _mp_face_mesh.process(crop_rgb)
                    if mesh_res.multi_face_landmarks:
                        lms = mesh_res.multi_face_landmarks[0]
                        # indices for approximate eye landmarks (outer/inner points)
                        left_eye_idx = [33, 133, 160, 159, 158, 157, 173]
                        right_eye_idx = [263, 362, 387, 386, 385, 384, 398]
                        fh, fw = face_crop.shape[:2]

                        def avg_point(idxs):
                            pts = []
                            for idx in idxs:
                                lm = lms.landmark[idx]
                                pts.append((lm.x * fw, lm.y * fh))
                            pts = np.array(pts)
                            return pts.mean(axis=0)

                        left_c = avg_point(left_eye_idx)
                        right_c = avg_point(right_eye_idx)
                        dx = right_c[0] - left_c[0]
                        dy = right_c[1] - left_c[1]
                        angle = np.degrees(np.arctan2(dy, dx))

                        # rotate to make eyes horizontal
                        center = (fw // 2, fh // 2)
                        M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # negative angle to deskew
                        rotated = cv2.warpAffine(face_crop, M, (fw, fh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

                        # after rotation, re-crop central square to avoid black borders
                        s = min(rotated.shape[0], rotated.shape[1])
                        cy2, cx2 = rotated.shape[0] // 2, rotated.shape[1] // 2
                        x1r = max(0, cx2 - s // 2)
                        y1r = max(0, cy2 - s // 2)
                        face_crop = rotated[y1r:y1r + s, x1r:x1r + s]
                except Exception:
                    # if face mesh fails, silently continue with unaligned crop
                    pass

            # ensure crop not empty
            if face_crop is None or face_crop.size == 0:
                return None
            return face_crop

    # fallback: Haar Cascade (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = haar.detectMultiScale(gray, 1.1, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        if w < min_face_size or h < min_face_size:
            return None
        side = max(w, h)
        pad = int(side * pad_ratio)
        cx = x + w // 2
        cy = y + h // 2
        half = int(side // 2)
        x1 = max(0, cx - half - pad)
        y1 = max(0, cy - half - pad)
        x2 = min(w_frame, cx + half + pad)
        y2 = min(h_frame, cy + half + pad)
        face_crop = frame[y1:y2, x1:x2].copy()
        return face_crop

    return None

# -------------------------------
# Preprocess face for model
# - target_size: tuple (H, W, C)
# - performs:
#    * square resizing
#    * grayscale conversion if model expects 1 channel
#    * CLAHE for contrast boosting (useful on low-light faces)
#    * normalization to [0,1]
# -------------------------------
def preprocess_face_for_model(face_bgr, target_size):
    H, W, C = target_size
    # protect against tiny crops
    if face_bgr is None or face_bgr.size == 0:
        raise ValueError("Empty face image in preprocess.")

    if C == 1:
        gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA if gray.shape[0] > H else cv2.INTER_CUBIC)
        # apply CLAHE to boost contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = gray.astype('float32') / 255.0
        gray = np.expand_dims(gray, axis=-1)
        return np.expand_dims(gray, axis=0)
    else:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA if rgb.shape[0] > H else cv2.INTER_CUBIC)
        rgb = rgb.astype('float32') / 255.0
        return np.expand_dims(rgb, axis=0)

# -------------------------------
# Smoother for video frames
# -------------------------------
class SoftmaxSmoother:
    def __init__(self, maxlen=5, num_classes=7):
        self.buf = deque(maxlen=maxlen)
        self.num_classes = num_classes

    def add(self, probs):
        self.buf.append(np.asarray(probs, dtype=np.float32))

    def get_average(self):
        if not self.buf:
            return None
        return np.mean(np.stack(list(self.buf)), axis=0)

# -------------------------------
# Predict emotion from a full frame (detects & crops face first)
# - model: Keras model
# - model_input_shape: (H,W,C) returned by load_fer_model
# - has_softmax: boolean indicating if final activation is softmax
# - returns (label, confidence, probs_array)
# -------------------------------
def predict_emotion_from_frame(model, model_input_shape, frame_bgr, smoother=None,
                               use_mediapipe=True, conf_threshold=0.45, has_softmax=False):
    face = detect_and_crop_face(frame_bgr, use_mediapipe=use_mediapipe, align=True)
    if face is None:
        return 'no_face', 0.0, None

    arr = preprocess_face_for_model(face, model_input_shape)
    preds = model.predict(arr, verbose=0)[0]

    # if model didn't output probabilities, apply softmax
    # Heuristic: if values are not in [0,1] or sum != 1 -> softmax
    if not has_softmax:
        if np.any(preds < 0) or not np.isclose(preds.sum(), 1.0, rtol=1e-2, atol=1e-3):
            preds = softmax(preds)

    if smoother is not None:
        smoother.add(preds)
        avg = smoother.get_average()
        if avg is not None:
            preds = avg

    idx = int(np.argmax(preds))
    conf = float(preds[idx])

    if conf < conf_threshold:
        return 'unknown', conf, preds
    return EMOTION_LABELS.get(idx, 'unknown'), conf, preds

# -------------------------------
# Utility: predict from a face-only image (if you already have a cropped face)
# -------------------------------
def predict_from_face_image(model, model_input_shape, face_bgr, smoother=None, has_softmax=False, conf_threshold=0.45):
    arr = preprocess_face_for_model(face_bgr, model_input_shape)
    preds = model.predict(arr, verbose=0)[0]
    if not has_softmax:
        if np.any(preds < 0) or not np.isclose(preds.sum(), 1.0, rtol=1e-2, atol=1e-3):
            preds = softmax(preds)
    if smoother is not None:
        smoother.add(preds)
        avg = smoother.get_average()
        if avg is not None:
            preds = avg
    idx = int(np.argmax(preds))
    conf = float(preds[idx])
    if conf < conf_threshold:
        return 'unknown', conf, preds
    return EMOTION_LABELS.get(idx, 'unknown'), conf, preds

# -------------------------------
# Example quick test helper
# -------------------------------
def demo_single_image(image_path, model, model_input_shape, has_softmax=False):
    img = cv2.imread(image_path)
    label, conf, probs = predict_emotion_from_frame(model, model_input_shape, img, smoother=None, has_softmax=has_softmax)
    print("Predicted:", label, " confidence:", conf)
    if probs is not None:
        print("Probs:", probs)
    # annotate and show
    try:
        face = detect_and_crop_face(img, align=True)
        if face is not None:
            annotated = img.copy()
            cv2.putText(annotated, f"{label} ({conf:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
            cv2.imshow("Annotated", annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    except Exception:
        pass

