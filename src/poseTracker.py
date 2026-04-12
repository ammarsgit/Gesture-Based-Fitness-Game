import time
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

srcDir = Path(__file__).resolve().parent
if str(srcDir) not in sys.path:
    sys.path.insert(0, str(srcDir))

from exercise_detection import ExerciseDetector
from game_logic import handle_event


windowName = "Bright Arm Workout Coach"
poseLandmark = vision.PoseLandmark

modelsDir = Path(__file__).resolve().parent.parent / "models"
modelQuality = "full"  # change to "lite" if needed
modelPath = modelsDir / f"pose_landmarker_{modelQuality}.task"

startFullscreen = False
displayMode = "cover"
cameraWidth = 960
cameraHeight = 540
inferenceWidth = 640
inferenceHeight = 360
inferenceStride = 1
fpsSmoothing = 0.2

poseConnections = vision.PoseLandmarksConnections.POSE_LANDMARKS
keyJoints = [
    poseLandmark.LEFT_SHOULDER,
    poseLandmark.RIGHT_SHOULDER,
    poseLandmark.LEFT_ELBOW,
    poseLandmark.RIGHT_ELBOW,
    poseLandmark.LEFT_WRIST,
    poseLandmark.RIGHT_WRIST,
]


def setFullscreen(isFullscreen):
    windowMode = cv2.WINDOW_FULLSCREEN if isFullscreen else cv2.WINDOW_NORMAL
    cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, windowMode)


def configureWindow():
    cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(windowName, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(windowName, 1180, 780)
    setFullscreen(startFullscreen)


def configureCamera(cap):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cameraWidth)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cameraHeight)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def fitFrameToWindow(frame):
    frameHeight, frameWidth = frame.shape[:2]

    try:
        _, _, windowWidth, windowHeight = cv2.getWindowImageRect(windowName)
    except cv2.error:
        return frame

    if windowWidth <= 0 or windowHeight <= 0:
        return frame

    if displayMode == "cover":
        scale = max(windowWidth / frameWidth, windowHeight / frameHeight)
        resizedWidth = max(1, int(frameWidth * scale))
        resizedHeight = max(1, int(frameHeight * scale))
        resizedFrame = cv2.resize(frame, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)

        offsetX = max(0, (resizedWidth - windowWidth) // 2)
        offsetY = max(0, (resizedHeight - windowHeight) // 2)
        return resizedFrame[offsetY:offsetY + windowHeight, offsetX:offsetX + windowWidth]

    scale = min(windowWidth / frameWidth, windowHeight / frameHeight)
    resizedWidth = max(1, int(frameWidth * scale))
    resizedHeight = max(1, int(frameHeight * scale))
    resizedFrame = cv2.resize(frame, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((windowHeight, windowWidth, 3), dtype=np.uint8)
    offsetX = (windowWidth - resizedWidth) // 2
    offsetY = (windowHeight - resizedHeight) // 2
    canvas[offsetY:offsetY + resizedHeight, offsetX:offsetX + resizedWidth] = resizedFrame
    return canvas


def drawPose(frame, landmarks):
    height, width, _ = frame.shape

    for connection in poseConnections:
        start_landmark = landmarks[connection.start]
        end_landmark = landmarks[connection.end]

        start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
        end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
        cv2.line(frame, start_point, end_point, (80, 255, 120), 3)

    for joint in keyJoints:
        landmark = landmarks[joint]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 10, (0, 255, 255), -1)
        cv2.circle(frame, (x, y), 14, (255, 80, 180), 2)


def createPoseLandmarker():
    if not modelPath.exists():
        raise FileNotFoundError(f"Pose model not found at {modelPath}")

    base_options = python.BaseOptions(model_asset_path=str(modelPath))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return vision.PoseLandmarker.create_from_options(options)


def createInferenceFrame(frame):
    frameHeight, frameWidth = frame.shape[:2]
    if frameWidth <= inferenceWidth and frameHeight <= inferenceHeight:
        return frame

    scale = min(inferenceWidth / frameWidth, inferenceHeight / frameHeight)
    resizedWidth = max(1, int(frameWidth * scale))
    resizedHeight = max(1, int(frameHeight * scale))
    return cv2.resize(frame, (resizedWidth, resizedHeight), interpolation=cv2.INTER_LINEAR)


def detectPose(landmarker, frame):
    inferenceFrame = createInferenceFrame(frame)
    rgbFrame = cv2.cvtColor(inferenceFrame, cv2.COLOR_BGR2RGB)
    mpImage = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbFrame)
    timestampMs = int(time.perf_counter() * 1000)
    return landmarker.detect_for_video(mpImage, timestampMs)


def getStatus(result):
    if result.pose_landmarks:
        return "PERSON DETECTED", (80, 255, 120), result.pose_landmarks[0]
    return "NO PERSON DETECTED", (70, 90, 255), None


def updateSmoothedFps(smoothedFps, currentTime, previousTime):
    instantFps = 1 / max(currentTime - previousTime, 1e-6)
    if smoothedFps == 0:
        return instantFps
    return (1 - fpsSmoothing) * smoothedFps + fpsSmoothing * instantFps


def draw_panel(frame, top_left, bottom_right, color, alpha=0.55, radius=18):
    overlay = frame.copy()
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_progress_bar(frame, x, y, w, h, progress):
    progress = max(0.0, min(1.0, progress))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
    fill_w = int(w * progress)
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), (0, 220, 255), -1)


def drawHud(frame, detector, statusText, statusColor, fps, eventText):
    frameHeight, frameWidth = frame.shape[:2]

    # Top banner
    draw_panel(frame, (20, 20), (frameWidth - 20, 120), (130, 40, 255), 0.45)
    cv2.putText(frame, "BRIGHT ARM WORKOUT COACH", (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, statusText, (40, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, statusColor, 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {int(fps)}", (frameWidth - 180, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

    # Left info card
    draw_panel(frame, (20, 140), (400, 360), (20, 170, 255), 0.42)
    cv2.putText(frame, detector.get_step_count_text(), (40, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detector.get_current_step_name(), (40, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frame, f"Time Left: {detector.remaining_seconds}s", (40, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    draw_progress_bar(frame, 40, 305, 320, 24, detector.progress)

    # Right feedback card
    draw_panel(frame, (frameWidth - 470, 140), (frameWidth - 20, 360), (255, 120, 40), 0.42)
    cv2.putText(frame, "LIVE FEEDBACK", (frameWidth - 440, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detector.current_feedback, (frameWidth - 440, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detector.last_debug_text, (frameWidth - 440, 295),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    # Bottom controls
    draw_panel(frame, (20, frameHeight - 95), (frameWidth - 20, frameHeight - 20), (40, 210, 120), 0.40)
    controls = "S = START    T = RESET    F = FULLSCREEN    C = FIT MODE    Q = QUIT"
    cv2.putText(frame, controls, (40, frameHeight - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

    if eventText:
        cv2.putText(frame, eventText, (40, frameHeight - 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2, cv2.LINE_AA)


def showFrame(frame, detector, statusText, statusColor, fps, eventText):
    displayFrame = fitFrameToWindow(frame)
    drawHud(displayFrame, detector, statusText, statusColor, fps, eventText)
    cv2.imshow(windowName, displayFrame)
    return cv2.waitKey(1) & 0xFF


def processKeyboardEvent(key, exerciseDetector):
    if key == ord("s"):
        exerciseDetector.request_start()
        return "Start requested"

    if key == ord("t"):
        exerciseDetector.restart()
        return "Workout reset"

    return ""


def main():
    global displayMode

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    configureCamera(cap)
    configureWindow()

    isFullscreen = startFullscreen
    previousTime = time.perf_counter()
    smoothedFps = 0.0
    frameIndex = 0
    lastResult = None

    exerciseDetector = ExerciseDetector()
    lastEventText = ""
    lastEventTime = 0.0

    with createPoseLandmarker() as pose:
        while True:
            success, frame = cap.read()
            if not success:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)

            if frameIndex % inferenceStride == 0 or lastResult is None:
                lastResult = detectPose(pose, frame)

            result = lastResult
            statusText, statusColor, landmarks = getStatus(result)

            if landmarks is not None:
                drawPose(frame, landmarks)
                events = exerciseDetector.update(landmarks)
            else:
                events = exerciseDetector.update(None)

            for event in events:
                handle_event(event)
                label = event["type"].replace("_", " ").title()
                confidence = event.get("confidence")
                if confidence is None:
                    lastEventText = f"{label}"
                else:
                    lastEventText = f"{label} ({confidence:.2f})"
                lastEventTime = time.perf_counter()

            currentTime = time.perf_counter()
            smoothedFps = updateSmoothedFps(smoothedFps, currentTime, previousTime)
            previousTime = currentTime
            frameIndex += 1

            if currentTime - lastEventTime > 1.5:
                lastEventText = ""

            key = showFrame(frame, exerciseDetector, statusText, statusColor, smoothedFps, lastEventText)

            manualEventText = processKeyboardEvent(key, exerciseDetector)
            if manualEventText:
                lastEventText = manualEventText
                lastEventTime = time.perf_counter()

            if key == ord("f"):
                isFullscreen = not isFullscreen
                setFullscreen(isFullscreen)
            elif key == ord("c"):
                displayMode = "contain" if displayMode == "cover" else "cover"
            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()