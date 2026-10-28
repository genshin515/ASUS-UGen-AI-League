"""
Desk Posture & Screen-Distance Guard  (Stage I laptop prototype)

Pipeline:  webcam -> YOLOv8-pose -> 2 geometric metrics -> debounce -> local notification
Everything runs locally. Planned Stage II port: same model family as
yolov8s_pose.hef on UGen300 (Hailo-10H); only load_model()/infer() change.

Metrics (frontal webcam proxies, normalised so no absolute cm values are needed):
  head_ratio : (shoulder_mid_y - ear_mid_y) / shoulder_width
               -> gets SMALLER when you slouch / drop your head forward
  eye_ratio  : eye_distance_px / frame_width
               -> gets LARGER when your face is closer to the screen
Both are compared against a personal baseline captured during calibration.

Usage:
  pip install numpy ultralytics opencv-python plyer
  python posture_guard.py                 # webcam 0
  python posture_guard.py --source demo.mp4
Keys: c = recalibrate (sit upright, normal distance), q = quit
"""
import argparse
import time

import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint indices
L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 1, 2, 3, 4, 5, 6

# ---- tunable parameters ----------------------------------------------------
KPT_CONF = 0.5        # min keypoint confidence
CALIB_SEC = 5.0       # seconds of good posture used as baseline
SLOUCH_RATIO = 0.85   # head_ratio < baseline * this  -> slouching
CLOSE_RATIO = 1.25    # eye_ratio  > baseline * this  -> too close
HOLD_SEC = 3.0        # must stay bad this long before alerting
COOLDOWN_SEC = 60.0   # min seconds between repeated alerts per issue
# ----------------------------------------------------------------------------


def load_model():
    # Stage II: replace with HailoRT load of yolov8s_pose.hef (hailo10h)
    return YOLO("yolov8n-pose.pt")


def infer(model, frame):
    """Return (17,3) keypoints [x, y, conf] of the largest person, or None."""
    res = model(frame, verbose=False)[0]
    if res.keypoints is None or len(res.boxes) == 0:
        return None
    areas = (res.boxes.xywh[:, 2] * res.boxes.xywh[:, 3]).cpu().numpy()
    return res.keypoints.data[int(areas.argmax())].cpu().numpy()


def compute_metrics(k, frame_w):
    """Return (head_ratio, eye_ratio) or None if keypoints are unreliable."""
    need = [L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH]
    if (k[need, 2] < KPT_CONF).any():
        return None
    shoulder_w = np.linalg.norm(k[L_SH, :2] - k[R_SH, :2])
    if shoulder_w < 1:
        return None
    ear_y = (k[L_EAR, 1] + k[R_EAR, 1]) / 2
    sh_y = (k[L_SH, 1] + k[R_SH, 1]) / 2
    head_ratio = (sh_y - ear_y) / shoulder_w
    eye_ratio = np.linalg.norm(k[L_EYE, :2] - k[R_EYE, :2]) / frame_w
    return head_ratio, eye_ratio


def notify(title, msg):
    print(f"[ALERT] {title}: {msg}")
    try:
        from plyer import notification
        notification.notify(title=title, message=msg, timeout=5)
    except Exception:
        pass  # on-screen banner + console are the fallback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=0, help="webcam index or video file")
    args = ap.parse_args()
    src = int(args.source) if str(args.source).isdigit() else args.source

    model = load_model()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit("Cannot open video source")

    calib_samples, baseline = [], None
    calib_start = time.time()
    bad_since = {"slouch": None, "close": None}
    last_alert = {"slouch": 0.0, "close": 0.0}
    banner, banner_until = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        h, w = frame.shape[:2]
        m = None
        k = infer(model, frame)
        if k is not None:
            m = compute_metrics(k, w)

        now = time.time()
        status = ""
        if baseline is None:
            status = f"Calibrating... sit upright ({max(0, CALIB_SEC - (now - calib_start)):.0f}s)"
            if m is not None:
                calib_samples.append(m)
            if now - calib_start >= CALIB_SEC and len(calib_samples) >= 10:
                baseline = np.median(np.array(calib_samples), axis=0)
                print(f"Baseline head_ratio={baseline[0]:.2f}, eye_ratio={baseline[1]:.3f}")
        elif m is not None:
            issues = {
                "slouch": m[0] < baseline[0] * SLOUCH_RATIO,
                "close": m[1] > baseline[1] * CLOSE_RATIO,
            }
            msgs = {
                "slouch": ("Posture", "You're slouching - sit up straight."),
                "close": ("Screen distance", "Too close - lean back from the screen."),
            }
            for name, bad in issues.items():
                if not bad:
                    bad_since[name] = None
                    continue
                bad_since[name] = bad_since[name] or now
                if now - bad_since[name] >= HOLD_SEC and now - last_alert[name] >= COOLDOWN_SEC:
                    notify(*msgs[name])
                    last_alert[name] = now
                    banner, banner_until = msgs[name][1], now + 4
            status = f"head {m[0]:.2f}/{baseline[0]:.2f}  eye {m[1]:.3f}/{baseline[1]:.3f}"
        else:
            status = "No reliable pose (face camera, show shoulders)"

        # ---- overlay (useful for the demo video) ----
        if k is not None:
            for x, y, c in k[[L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH]]:
                if c >= KPT_CONF:
                    cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
        cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        fps = 1.0 / max(time.time() - t0, 1e-6)
        cv2.putText(frame, f"{fps:.0f} FPS | local only", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        if now < banner_until:
            cv2.rectangle(frame, (0, 40), (w, 85), (0, 0, 200), -1)
            cv2.putText(frame, banner, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Posture Guard", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            baseline, calib_samples, calib_start = None, [], time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
