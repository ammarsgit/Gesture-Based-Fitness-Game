import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from exercise_detection import ExerciseDetector
from game_logic import handle_event


windowName = "Gesture Fitness Game"
poseLandmark = mp.solutions.pose.PoseLandmark
modelsDir = Path(__file__).resolve().parent / "models"
modelQuality = "lite"
modelPath = modelsDir / f"pose_landmarker_{modelQuality}.task"
startFullscreen = False
displayMode = "cover"
cameraWidth = 960
cameraHeight = 540
inferenceWidth = 640
inferenceHeight = 360
inferenceStride = 2
fpsSmoothing = 0.2
poseConnections = mp.solutions.pose.POSE_CONNECTIONS
keyJoints = [
	poseLandmark.LEFT_SHOULDER,
	poseLandmark.RIGHT_SHOULDER,
	poseLandmark.LEFT_ELBOW,
	poseLandmark.RIGHT_ELBOW,
	poseLandmark.LEFT_WRIST,
	poseLandmark.RIGHT_WRIST,
	poseLandmark.LEFT_KNEE,
	poseLandmark.RIGHT_KNEE,
	poseLandmark.LEFT_ANKLE,
	poseLandmark.RIGHT_ANKLE,
	poseLandmark.LEFT_FOOT_INDEX,
	poseLandmark.RIGHT_FOOT_INDEX,
]


def setFullscreen(isFullscreen):
	windowMode = cv2.WINDOW_FULLSCREEN if isFullscreen else cv2.WINDOW_NORMAL
	cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, windowMode)


def configureWindow():
	cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
	cv2.setWindowProperty(windowName, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
	cv2.resizeWindow(windowName, 960, 720)
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
		start_landmark = landmarks[connection[0]]
		end_landmark = landmarks[connection[1]]

		start_point = (int(start_landmark.x * width), int(start_landmark.y * height))
		end_point = (int(end_landmark.x * width), int(end_landmark.y * height))
		cv2.line(frame, start_point, end_point, (0, 255, 0), 2)

	for joint in keyJoints:
		landmark = landmarks[joint]
		x = int(landmark.x * width)
		y = int(landmark.y * height)
		cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
		cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)


def createPoseLandmarker():
	if not modelPath.exists():
		raise FileNotFoundError(f"Pose model not found at {modelPath}")

	base_options = python.BaseOptions(model_asset_path=str(modelPath))
	options = vision.PoseLandmarkerOptions(
		base_options=base_options,
		running_mode=vision.RunningMode.VIDEO,
		num_poses=1,
		min_pose_detection_confidence=0.5,
		min_pose_presence_confidence=0.5,
		min_tracking_confidence=0.5,
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
		return "Person Detected", (0, 255, 0), result.pose_landmarks[0]

	return "No Person Detected", (0, 0, 255), None


def updateSmoothedFps(smoothedFps, currentTime, previousTime):
	instantFps = 1 / max(currentTime - previousTime, 1e-6)
	if smoothedFps == 0:
		return instantFps

	return (1 - fpsSmoothing) * smoothedFps + fpsSmoothing * instantFps


def drawHud(frame, statusText, statusColor, fps, exerciseText, eventText):
	frameHeight, frameWidth = frame.shape[:2]
	marginX = max(20, frameWidth // 40)
	marginY = max(40, frameHeight // 18)
	fontScale = max(0.8, min(frameWidth, frameHeight) / 900)
	lineThickness = max(2, int(fontScale * 2))
	lineGap = max(34, int(44 * fontScale))

	cv2.putText(
		frame,
		statusText,
		(marginX, marginY),
		cv2.FONT_HERSHEY_SIMPLEX,
		fontScale,
		statusColor,
		lineThickness,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		f"FPS: {int(fps)}",
		(marginX, marginY + lineGap),
		cv2.FONT_HERSHEY_SIMPLEX,
		fontScale * 0.85,
		(255, 255, 0),
		lineThickness,
		cv2.LINE_AA,
	)
	cv2.putText(
		frame,
		exerciseText,
		(marginX, marginY + lineGap * 2),
		cv2.FONT_HERSHEY_SIMPLEX,
		fontScale * 0.65,
		(255, 255, 255),
		max(1, lineThickness - 1),
		cv2.LINE_AA,
	)
	if eventText:
		cv2.putText(
			frame,
			eventText,
			(marginX, marginY + lineGap * 3),
			cv2.FONT_HERSHEY_SIMPLEX,
			fontScale * 0.7,
			(0, 200, 255),
			max(1, lineThickness - 1),
			cv2.LINE_AA,
		)


def showFrame(frame, statusText, statusColor, fps, exerciseText, eventText):
	displayFrame = fitFrameToWindow(frame)
	drawHud(displayFrame, statusText, statusColor, fps, exerciseText, eventText)
	cv2.imshow(windowName, displayFrame)
	key = cv2.waitKey(1) & 0xFF
	return key


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
			exerciseText = exerciseDetector.last_debug_text

			if landmarks is not None:
				drawPose(frame, landmarks)
				for event in exerciseDetector.update(landmarks):
					handle_event(event)
					label = "Push-up" if event["type"] == "pushup" else "Squat"
					lastEventText = f"Detected {label} ({event['confidence']:.2f})"
					lastEventTime = time.perf_counter()
				exerciseText = exerciseDetector.last_debug_text
			else:
				exerciseDetector.update(None)
				exerciseText = exerciseDetector.last_debug_text

			currentTime = time.perf_counter()
			smoothedFps = updateSmoothedFps(smoothedFps, currentTime, previousTime)
			previousTime = currentTime
			frameIndex += 1
			if currentTime - lastEventTime > 1.5:
				lastEventText = ""

			key = showFrame(frame, statusText, statusColor, smoothedFps, exerciseText, lastEventText)
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