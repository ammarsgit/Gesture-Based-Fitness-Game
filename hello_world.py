import time

import cv2
import mediapipe as mp


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

KEY_JOINTS = [
	mp_pose.PoseLandmark.LEFT_SHOULDER,
	mp_pose.PoseLandmark.RIGHT_SHOULDER,
	mp_pose.PoseLandmark.LEFT_ELBOW,
	mp_pose.PoseLandmark.RIGHT_ELBOW,
	mp_pose.PoseLandmark.LEFT_WRIST,
	mp_pose.PoseLandmark.RIGHT_WRIST,
]


def draw_key_joints(frame, landmarks):
	height, width, _ = frame.shape

	for joint in KEY_JOINTS:
		landmark = landmarks.landmark[joint]
		x = int(landmark.x * width)
		y = int(landmark.y * height)
		cv2.circle(frame, (x, y), 8, (0, 255, 255), -1)
		cv2.circle(frame, (x, y), 12, (0, 0, 0), 2)


def main():
	cap = cv2.VideoCapture(0)

	if not cap.isOpened():
		print("Could not open webcam.")
		return

	prev_time = time.perf_counter()

	with mp_pose.Pose(
		min_detection_confidence=0.5,
		min_tracking_confidence=0.5,
		model_complexity=1,
	) as pose:
		while True:
			success, frame = cap.read()
			if not success:
				print("Failed to read frame from webcam.")
				break

			frame = cv2.flip(frame, 1)

			rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			rgb_frame.flags.writeable = False
			results = pose.process(rgb_frame)
			rgb_frame.flags.writeable = True

			person_detected = results.pose_landmarks is not None

			if person_detected:
				mp_drawing.draw_landmarks(
					frame,
					results.pose_landmarks,
					mp_pose.POSE_CONNECTIONS,
					mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
					mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=2),
				)
				draw_key_joints(frame, results.pose_landmarks)
				status_text = "Person Detected"
				status_color = (0, 255, 0)
			else:
				status_text = "No Person Detected"
				status_color = (0, 0, 255)

			current_time = time.perf_counter()
			fps = 1 / max(current_time - prev_time, 1e-6)
			prev_time = current_time

			cv2.putText(
				frame,
				status_text,
				(20, 40),
				cv2.FONT_HERSHEY_SIMPLEX,
				1,
				status_color,
				2,
				cv2.LINE_AA,
			)
			cv2.putText(
				frame,
				f"FPS: {int(fps)}",
				(20, 80),
				cv2.FONT_HERSHEY_SIMPLEX,
				0.8,
				(255, 255, 0),
				2,
				cv2.LINE_AA,
			)

			cv2.imshow("Gesture Fitness Game", frame)

			# Press Q to quit.
			if cv2.waitKey(1) & 0xFF == ord("q"):
				break

	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()