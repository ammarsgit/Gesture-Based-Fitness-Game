import cv2
import time
import sys
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

srcDir = Path(__file__).resolve().parent
if str(srcDir) not in sys.path:
    sys.path.insert(0, str(srcDir))

from exercise_detection import ExerciseDetector
from game_logic import handle_event

poseLandmark = vision.PoseLandmark
modelPath = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_lite.task"
windowName = "GymPossible"


def create_detector():
    base = python.BaseOptions(model_asset_path=str(modelPath))
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def setup_window():
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(windowName, 1280, 820)


def draw_panel(frame, top_left, bottom_right, color, alpha=0.32):
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_progress_bar(frame, x, y, w, h, progress):
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), (0, 220, 255), -1)


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    connections = vision.PoseLandmarksConnections.POSE_LANDMARKS

    for connection in connections:
        start_landmark = landmarks[connection.start]
        end_landmark = landmarks[connection.end]
        start_point = (int(start_landmark.x * w), int(start_landmark.y * h))
        end_point = (int(end_landmark.x * w), int(end_landmark.y * h))
        cv2.line(frame, start_point, end_point, (80, 255, 120), 3)

    joints = [
        poseLandmark.LEFT_SHOULDER,
        poseLandmark.RIGHT_SHOULDER,
        poseLandmark.LEFT_ELBOW,
        poseLandmark.RIGHT_ELBOW,
        poseLandmark.LEFT_WRIST,
        poseLandmark.RIGHT_WRIST,
    ]

    for joint in joints:
        landmark = landmarks[joint]
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(frame, (x, y), 12, (255, 80, 180), 2)


def draw_dashboard(frame, detector, fps, event_text, person_detected):
    h, w = frame.shape[:2]

    # Top banner
    draw_panel(frame, (28, 20), (w - 28, 102), (170, 70, 180), 0.34)
    cv2.putText(
        frame,
        "GYMPOSSIBLE",
        (48, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    status_text = "PERSON DETECTED" if person_detected else "NO PERSON DETECTED"
    status_color = (120, 255, 120) if person_detected else (90, 120, 255)

    cv2.putText(
        frame,
        status_text,
        (48, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (w - 165, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Spread-out cards like the better screenshot
    left_x1, left_y1 = 60, 160
    left_x2, left_y2 = 370, 380

    right_x1, right_y1 = w - 430, 160
    right_x2, right_y2 = w - 60, 380

    # Left card
    draw_panel(frame, (left_x1, left_y1), (left_x2, left_y2), (232, 176, 74), 0.30)
    cv2.putText(
        frame,
        detector.get_step_count_text(),
        (left_x1 + 18, left_y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        detector.get_current_step_name(),
        (left_x1 + 18, left_y1 + 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Time Left: {detector.remaining_seconds}s",
        (left_x1 + 18, left_y1 + 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    draw_progress_bar(frame, left_x1 + 18, left_y1 + 164, 250, 18, detector.progress)

    # Right card
    draw_panel(frame, (right_x1, right_y1), (right_x2, right_y2), (80, 130, 225), 0.30)
    cv2.putText(
        frame,
        "LIVE FEEDBACK",
        (right_x1 + 18, right_y1 + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    feedback_color = (255, 255, 255)
    if detector.current_feedback_kind == "good":
        feedback_color = (120, 255, 120)
    elif detector.current_feedback_kind == "correction":
        feedback_color = (80, 230, 255)

    cv2.putText(
        frame,
        detector.current_feedback,
        (right_x1 + 18, right_y1 + 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        feedback_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        detector.status_text,
        (right_x1 + 18, right_y1 + 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        detector.last_debug_text,
        (right_x1 + 18, right_y1 + 182),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Bottom controls
    draw_panel(frame, (28, h - 92), (w - 28, h - 24), (70, 180, 90), 0.34)
    cv2.putText(
        frame,
        "S = START    R = RESET    Q = QUIT",
        (48, h - 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    if event_text:
        cv2.putText(
            frame,
            event_text,
            (48, h - 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def main():
    if not modelPath.exists():
        print(f"Missing model file: {modelPath}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    setup_window()
    detector = ExerciseDetector()

    last_event_text = ""
    last_event_time = 0.0

    previous_time = time.perf_counter()
    smoothed_fps = 0.0

    with create_detector() as pose:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = pose.detect_for_video(mp_img, int(time.perf_counter() * 1000))
            landmarks = result.pose_landmarks[0] if result.pose_landmarks else None

            if landmarks is not None:
                draw_pose(frame, landmarks)

            events = detector.update(landmarks)

            for event in events:
                handle_event(event)
                last_event_text = event["type"].replace("_", " ").title()
                last_event_time = time.perf_counter()

            current_time = time.perf_counter()
            instant_fps = 1 / max(current_time - previous_time, 1e-6)
            if smoothed_fps == 0:
                smoothed_fps = instant_fps
            else:
                smoothed_fps = 0.8 * smoothed_fps + 0.2 * instant_fps
            previous_time = current_time

            if current_time - last_event_time > 1.5:
                last_event_text = ""

            draw_dashboard(frame, detector, smoothed_fps, last_event_text, landmarks is not None)
            cv2.imshow(windowName, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                detector.request_start()
            elif key == ord("r"):
                detector.restart()
                last_event_text = "Workout reset"
                last_event_time = time.perf_counter()
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()