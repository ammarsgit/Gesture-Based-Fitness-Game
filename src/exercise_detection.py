import math
import time

from mediapipe.tasks.python import vision


poseLandmark = vision.PoseLandmark


def clamp(value, minimum, maximum):
	return max(minimum, min(maximum, value))


def average(values):
	return sum(values) / len(values)


def point_xy(landmark):
	return landmark.x, landmark.y


def point_distance(first, second):
	first_x, first_y = point_xy(first)
	second_x, second_y = point_xy(second)
	return math.hypot(first_x - second_x, first_y - second_y)


def joint_angle(first, middle, last):
	ax, ay = point_xy(first)
	bx, by = point_xy(middle)
	cx, cy = point_xy(last)

	first_vector = (ax - bx, ay - by)
	last_vector = (cx - bx, cy - by)

	first_length = math.hypot(*first_vector)
	last_length = math.hypot(*last_vector)
	if first_length == 0 or last_length == 0:
		return 180.0

	dot_product = first_vector[0] * last_vector[0] + first_vector[1] * last_vector[1]
	cosine = clamp(dot_product / (first_length * last_length), -1.0, 1.0)
	return math.degrees(math.acos(cosine))


class ExerciseDetector:
	def __init__(self):
		self.squat_stage = "up"
		self.pushup_stage = "up"
		self.plank_stage = "not_ready"
		self.plank_started_at = None
		self.left_tap_active = False
		self.right_tap_active = False
		self.deepest_squat_angle = 180.0
		self.deepest_pushup_angle = 180.0
		self.last_debug_text = "Exercise: waiting for movement"

	def update(self, landmarks):
		if landmarks is None:
			self.last_debug_text = "Exercise: no landmarks"
			return []

		events = []
		squat_event, squat_debug = self._detect_squat(landmarks)
		pushup_event, pushup_debug = self._detect_pushup(landmarks)
		plank_event, plank_debug, plank_ready = self._detect_plank(landmarks)
		shoulder_tap_event, shoulder_tap_debug = self._detect_shoulder_tap(landmarks, plank_ready)

		if squat_event is not None:
			events.append(squat_event)
		if pushup_event is not None:
			events.append(pushup_event)
		if plank_event is not None:
			events.append(plank_event)
		if shoulder_tap_event is not None:
			events.append(shoulder_tap_event)

		self.last_debug_text = f"{squat_debug} | {pushup_debug} | {plank_debug} | {shoulder_tap_debug}"
		return events

	def _body_alignment(self, landmarks):
		shoulder_y = average([
			landmarks[poseLandmark.LEFT_SHOULDER].y,
			landmarks[poseLandmark.RIGHT_SHOULDER].y,
		])
		hip_y = average([
			landmarks[poseLandmark.LEFT_HIP].y,
			landmarks[poseLandmark.RIGHT_HIP].y,
		])
		ankle_y = average([
			landmarks[poseLandmark.LEFT_ANKLE].y,
			landmarks[poseLandmark.RIGHT_ANKLE].y,
		])

		shoulder_hip_gap = abs(shoulder_y - hip_y)
		hip_ankle_gap = abs(hip_y - ankle_y)
		body_is_horizontal = shoulder_hip_gap < 0.16 and hip_ankle_gap < 0.18
		return body_is_horizontal, shoulder_hip_gap, hip_ankle_gap

	def _detect_squat(self, landmarks):
		left_knee_angle = joint_angle(
			landmarks[poseLandmark.LEFT_HIP],
			landmarks[poseLandmark.LEFT_KNEE],
			landmarks[poseLandmark.LEFT_ANKLE],
		)
		right_knee_angle = joint_angle(
			landmarks[poseLandmark.RIGHT_HIP],
			landmarks[poseLandmark.RIGHT_KNEE],
			landmarks[poseLandmark.RIGHT_ANKLE],
		)
		knee_angle = average([left_knee_angle, right_knee_angle])

		if self.squat_stage == "up" and knee_angle < 105:
			self.squat_stage = "down"
			self.deepest_squat_angle = knee_angle
		elif self.squat_stage == "down":
			self.deepest_squat_angle = min(self.deepest_squat_angle, knee_angle)
			if knee_angle > 155:
				confidence = clamp((165 - self.deepest_squat_angle) / 70, 0.0, 1.0)
				self.squat_stage = "up"
				self.deepest_squat_angle = 180.0
				return {"type": "squat", "confidence": confidence}, f"Squat: rep ({int(knee_angle)} deg)"

		return None, f"Squat: {self.squat_stage} ({int(knee_angle)} deg)"

	def _detect_pushup(self, landmarks):
		left_elbow_angle = joint_angle(
			landmarks[poseLandmark.LEFT_SHOULDER],
			landmarks[poseLandmark.LEFT_ELBOW],
			landmarks[poseLandmark.LEFT_WRIST],
		)
		right_elbow_angle = joint_angle(
			landmarks[poseLandmark.RIGHT_SHOULDER],
			landmarks[poseLandmark.RIGHT_ELBOW],
			landmarks[poseLandmark.RIGHT_WRIST],
		)
		elbow_angle = average([left_elbow_angle, right_elbow_angle])
		body_is_horizontal, shoulder_hip_gap, hip_ankle_gap = self._body_alignment(landmarks)

		if body_is_horizontal and self.pushup_stage == "up" and elbow_angle < 95:
			self.pushup_stage = "down"
			self.deepest_pushup_angle = elbow_angle
		elif self.pushup_stage == "down":
			self.deepest_pushup_angle = min(self.deepest_pushup_angle, elbow_angle)
			if elbow_angle > 150 and body_is_horizontal:
				bend_score = clamp((160 - self.deepest_pushup_angle) / 75, 0.0, 1.0)
				alignment_score = clamp(1.0 - ((shoulder_hip_gap / 0.16) + (hip_ankle_gap / 0.18)) / 2, 0.0, 1.0)
				confidence = clamp(0.65 * bend_score + 0.35 * alignment_score, 0.0, 1.0)
				self.pushup_stage = "up"
				self.deepest_pushup_angle = 180.0
				return {"type": "pushup", "confidence": confidence}, f"Push-up: rep ({int(elbow_angle)} deg)"
			if not body_is_horizontal and elbow_angle > 130:
				self.pushup_stage = "up"
				self.deepest_pushup_angle = 180.0

		mode_text = "ready" if body_is_horizontal else "not ready"
		return None, f"Push-up: {self.pushup_stage}, {mode_text} ({int(elbow_angle)} deg)"

	def _detect_plank(self, landmarks):
		left_elbow_angle = joint_angle(
			landmarks[poseLandmark.LEFT_SHOULDER],
			landmarks[poseLandmark.LEFT_ELBOW],
			landmarks[poseLandmark.LEFT_WRIST],
		)
		right_elbow_angle = joint_angle(
			landmarks[poseLandmark.RIGHT_SHOULDER],
			landmarks[poseLandmark.RIGHT_ELBOW],
			landmarks[poseLandmark.RIGHT_WRIST],
		)
		elbow_angle = average([left_elbow_angle, right_elbow_angle])
		body_is_horizontal, shoulder_hip_gap, hip_ankle_gap = self._body_alignment(landmarks)
		in_plank = body_is_horizontal and elbow_angle > 150

		if in_plank:
			if self.plank_started_at is None:
				self.plank_started_at = time.perf_counter()
				self.plank_stage = "holding"
			hold_time = time.perf_counter() - self.plank_started_at
			if self.plank_stage == "holding" and hold_time >= 1.5:
				alignment_score = clamp(1.0 - ((shoulder_hip_gap / 0.16) + (hip_ankle_gap / 0.18)) / 2, 0.0, 1.0)
				arm_score = clamp((elbow_angle - 150) / 20, 0.0, 1.0)
				confidence = clamp(0.7 * alignment_score + 0.3 * arm_score, 0.0, 1.0)
				self.plank_stage = "counted"
				return {"type": "plank", "confidence": confidence}, f"Plank: held {hold_time:.1f}s", True
			return None, f"Plank: holding {hold_time:.1f}s", True

		self.plank_stage = "not_ready"
		self.plank_started_at = None
		self.left_tap_active = False
		self.right_tap_active = False
		return None, "Plank: not ready", False

	def _detect_shoulder_tap(self, landmarks, plank_ready):
		if not plank_ready:
			return None, "Shoulder Tap: not ready"

		left_wrist = landmarks[poseLandmark.LEFT_WRIST]
		right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
		left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
		right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]

		left_cross_distance = point_distance(left_wrist, right_shoulder)
		right_cross_distance = point_distance(right_wrist, left_shoulder)
		left_reset_distance = point_distance(left_wrist, left_shoulder)
		right_reset_distance = point_distance(right_wrist, right_shoulder)

		left_touching = left_cross_distance < 0.14 and left_wrist.y < left_shoulder.y + 0.08
		right_touching = right_cross_distance < 0.14 and right_wrist.y < right_shoulder.y + 0.08

		if left_touching and not self.left_tap_active:
			self.left_tap_active = True
			confidence = clamp((0.14 - left_cross_distance) / 0.10, 0.0, 1.0)
			return {"type": "shoulder_tap", "confidence": confidence}, "Shoulder Tap: left"
		if right_touching and not self.right_tap_active:
			self.right_tap_active = True
			confidence = clamp((0.14 - right_cross_distance) / 0.10, 0.0, 1.0)
			return {"type": "shoulder_tap", "confidence": confidence}, "Shoulder Tap: right"

		if self.left_tap_active and left_reset_distance > 0.18:
			self.left_tap_active = False
		if self.right_tap_active and right_reset_distance > 0.18:
			self.right_tap_active = False

		if left_touching:
			return None, "Shoulder Tap: left contact"
		if right_touching:
			return None, "Shoulder Tap: right contact"
		return None, "Shoulder Tap: ready"