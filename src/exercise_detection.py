import math
import time

from mediapipe.tasks.python import vision


poseLandmark = vision.PoseLandmark


def clamp(value, minimum, maximum):
	return max(minimum, min(maximum, value))


ANGLE_EMA_ALPHA = 0.35
MIN_LANDMARK_CONFIDENCE = 0.5
CALIBRATION_SECONDS = 3.0
CALIBRATION_MIN_FRAMES = 30
INACTIVITY_RESET_SECONDS = 2.5
SQUAT_REP_COOLDOWN = 0.45
PUSHUP_REP_COOLDOWN = 0.55
MIN_BODY_LINE_ANGLE = 145.0


def average(values):
	return sum(values) / len(values)


def point_xy(landmark):
	return landmark.x, landmark.y


def point_xyz(landmark):
	return landmark.x, landmark.y, landmark.z


def point_distance(first, second):
	first_x, first_y = point_xy(first)
	second_x, second_y = point_xy(second)
	return math.hypot(first_x - second_x, first_y - second_y)


def vertical_angle(upper, lower):
	dx = lower.x - upper.x
	dy = upper.y - lower.y
	return math.degrees(math.atan2(abs(dx), max(dy, 1e-6)))


def joint_angle(first, middle, last):
	ax, ay, az = point_xyz(first)
	bx, by, bz = point_xyz(middle)
	cx, cy, cz = point_xyz(last)

	first_vector = (ax - bx, ay - by, az - bz)
	last_vector = (cx - bx, cy - by, cz - bz)

	first_length = math.sqrt(sum(component * component for component in first_vector))
	last_length = math.sqrt(sum(component * component for component in last_vector))
	if first_length == 0 or last_length == 0:
		return 180.0

	dot_product = sum(first_vector[index] * last_vector[index] for index in range(3))
	cosine = clamp(dot_product / (first_length * last_length), -1.0, 1.0)
	return math.degrees(math.acos(cosine))


class ExerciseDetector:
	def __init__(self):
		self.calibration_started_at = time.perf_counter()
		self.calibration_progress = 0.0
		self.is_calibrating = True
		self.calibration_samples = {
			"squat_top_knee": [],
			"pushup_top_elbow": [],
			"body_line": [],
		}

		self.squat_stage = "top"
		self.pushup_stage = "top"
		self.squat_reached_depth = False
		self.pushup_reached_depth = False
		self.deepest_squat_angle = 180.0
		self.deepest_pushup_angle = 180.0
		self.last_rep_times = {"squat": 0.0, "pushup": 0.0}
		self.last_motion_time = time.perf_counter()
		self.filtered_angles = {}

		self.squat_down_angle = 120.0
		self.squat_up_angle = 150.0
		self.squat_depth_angle = 100.0
		self.pushup_down_angle = 125.0
		self.pushup_up_angle = 145.0
		self.pushup_depth_angle = 105.0
		self.body_line_min_angle = 160.0

		self.last_debug_text = "Calibration: hold neutral pose"

	def restart_calibration(self):
		self.calibration_started_at = time.perf_counter()
		self.calibration_progress = 0.0
		self.is_calibrating = True
		self.calibration_samples = {
			"squat_top_knee": [],
			"pushup_top_elbow": [],
			"body_line": [],
		}
		self.squat_stage = "top"
		self.pushup_stage = "top"
		self.squat_reached_depth = False
		self.pushup_reached_depth = False
		self.deepest_squat_angle = 180.0
		self.deepest_pushup_angle = 180.0
		self.filtered_angles = {}
		self.last_motion_time = time.perf_counter()
		self.last_debug_text = "Calibration: hold neutral pose"

	def update(self, landmarks):
		if landmarks is None:
			self.last_debug_text = "Exercise: no landmarks"
			return []

		if self.is_calibrating:
			self._update_calibration(landmarks)
			return []

		if not self._has_quality(landmarks, self._required_landmarks()):
			self.last_debug_text = "Exercise: low landmark confidence"
			return []

		events = []
		squat_event, squat_debug = self._detect_squat(landmarks)
		pushup_event, pushup_debug = self._detect_pushup(landmarks)

		if squat_event is not None:
			events.append(squat_event)
		if pushup_event is not None:
			events.append(pushup_event)

		now = time.perf_counter()
		if now - self.last_motion_time > INACTIVITY_RESET_SECONDS:
			self._reset_stages()

		self.last_debug_text = f"{squat_debug} | {pushup_debug}"
		return events

	def _required_landmarks(self):
		return [
			poseLandmark.LEFT_SHOULDER,
			poseLandmark.RIGHT_SHOULDER,
			poseLandmark.LEFT_ELBOW,
			poseLandmark.RIGHT_ELBOW,
			poseLandmark.LEFT_WRIST,
			poseLandmark.RIGHT_WRIST,
			poseLandmark.LEFT_HIP,
			poseLandmark.RIGHT_HIP,
			poseLandmark.LEFT_KNEE,
			poseLandmark.RIGHT_KNEE,
			poseLandmark.LEFT_ANKLE,
			poseLandmark.RIGHT_ANKLE,
		]

	def _landmark_confidence(self, landmark):
		visibility = getattr(landmark, "visibility", 1.0)
		presence = getattr(landmark, "presence", 1.0)
		return min(visibility, presence)

	def _has_quality(self, landmarks, required):
		for landmark_index in required:
			if self._landmark_confidence(landmarks[landmark_index]) < MIN_LANDMARK_CONFIDENCE:
				return False
		return True

	def _ema(self, key, value):
		previous = self.filtered_angles.get(key)
		if previous is None:
			self.filtered_angles[key] = value
			return value
		filtered_value = (1 - ANGLE_EMA_ALPHA) * previous + ANGLE_EMA_ALPHA * value
		self.filtered_angles[key] = filtered_value
		return filtered_value

	def _side_score(self, landmarks, side):
		if side == "left":
			indices = [
				poseLandmark.LEFT_SHOULDER,
				poseLandmark.LEFT_ELBOW,
				poseLandmark.LEFT_WRIST,
				poseLandmark.LEFT_HIP,
				poseLandmark.LEFT_KNEE,
				poseLandmark.LEFT_ANKLE,
			]
		else:
			indices = [
				poseLandmark.RIGHT_SHOULDER,
				poseLandmark.RIGHT_ELBOW,
				poseLandmark.RIGHT_WRIST,
				poseLandmark.RIGHT_HIP,
				poseLandmark.RIGHT_KNEE,
				poseLandmark.RIGHT_ANKLE,
			]
		return average([self._landmark_confidence(landmarks[index]) for index in indices])

	def _active_side(self, landmarks):
		left_score = self._side_score(landmarks, "left")
		right_score = self._side_score(landmarks, "right")
		return "left" if left_score >= right_score else "right"

	def _collect_calibration(self, landmarks):
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
		knee_top = average([left_knee_angle, right_knee_angle])

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
		elbow_top = average([left_elbow_angle, right_elbow_angle])

		left_body_line = joint_angle(
			landmarks[poseLandmark.LEFT_SHOULDER],
			landmarks[poseLandmark.LEFT_HIP],
			landmarks[poseLandmark.LEFT_ANKLE],
		)
		right_body_line = joint_angle(
			landmarks[poseLandmark.RIGHT_SHOULDER],
			landmarks[poseLandmark.RIGHT_HIP],
			landmarks[poseLandmark.RIGHT_ANKLE],
		)
		body_line = average([left_body_line, right_body_line])

		self.calibration_samples["squat_top_knee"].append(knee_top)
		self.calibration_samples["pushup_top_elbow"].append(elbow_top)
		self.calibration_samples["body_line"].append(body_line)

	def _finalize_calibration(self):
		squat_top = average(self.calibration_samples["squat_top_knee"])
		elbow_top = average(self.calibration_samples["pushup_top_elbow"])
		body_line = average(self.calibration_samples["body_line"])

		self.squat_down_angle = clamp(squat_top - 50.0, 100.0, 132.0)
		self.squat_up_angle = clamp(squat_top - 22.0, 135.0, 170.0)
		self.squat_depth_angle = clamp(squat_top - 78.0, 78.0, 112.0)

		self.pushup_down_angle = clamp(elbow_top - 38.0, 100.0, 132.0)
		self.pushup_up_angle = clamp(elbow_top - 16.0, 126.0, 170.0)
		self.pushup_depth_angle = clamp(elbow_top - 58.0, 86.0, 122.0)
		self.body_line_min_angle = clamp(body_line - 24.0, MIN_BODY_LINE_ANGLE, 174.0)

		self.is_calibrating = False
		self.calibration_progress = 1.0
		self.last_debug_text = "Exercise: ready for squat and push-up"

	def _update_calibration(self, landmarks):
		elapsed = time.perf_counter() - self.calibration_started_at
		self.calibration_progress = clamp(elapsed / CALIBRATION_SECONDS, 0.0, 1.0)

		if self._has_quality(landmarks, self._required_landmarks()):
			self._collect_calibration(landmarks)

		sample_count = len(self.calibration_samples["squat_top_knee"])
		if elapsed >= CALIBRATION_SECONDS and sample_count >= CALIBRATION_MIN_FRAMES:
			self._finalize_calibration()
			return

		seconds_left = max(0.0, CALIBRATION_SECONDS - elapsed)
		self.last_debug_text = f"Calibration: hold still {seconds_left:.1f}s"

	def _reset_stages(self):
		self.squat_stage = "top"
		self.pushup_stage = "top"
		self.squat_reached_depth = False
		self.pushup_reached_depth = False
		self.deepest_squat_angle = 180.0
		self.deepest_pushup_angle = 180.0

	def _body_alignment(self, landmarks, side):
		if side == "left":
			shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
			hip = landmarks[poseLandmark.LEFT_HIP]
			ankle = landmarks[poseLandmark.LEFT_ANKLE]
		else:
			shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
			hip = landmarks[poseLandmark.RIGHT_HIP]
			ankle = landmarks[poseLandmark.RIGHT_ANKLE]

		shoulder_hip_gap = abs(shoulder.y - hip.y)
		hip_ankle_gap = abs(hip.y - ankle.y)
		body_line_angle = joint_angle(shoulder, hip, ankle)
		body_is_horizontal = shoulder_hip_gap < 0.24 and hip_ankle_gap < 0.30
		line_is_strong = body_line_angle > self.body_line_min_angle
		return body_is_horizontal and line_is_strong, shoulder_hip_gap, hip_ankle_gap, body_line_angle

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
		knee_angle = self._ema("squat_knee", average([left_knee_angle, right_knee_angle]))

		now = time.perf_counter()
		if self.squat_stage == "top" and knee_angle < self.squat_down_angle:
			self.squat_stage = "descending"
			self.squat_reached_depth = False
			self.deepest_squat_angle = knee_angle
			self.last_motion_time = now
		elif self.squat_stage == "descending":
			self.deepest_squat_angle = min(self.deepest_squat_angle, knee_angle)
			if knee_angle < self.squat_depth_angle:
				self.squat_reached_depth = True
				self.last_motion_time = now
			if knee_angle > self.squat_up_angle:
				depth_score = clamp((self.squat_down_angle - self.deepest_squat_angle) / max(self.squat_down_angle - self.squat_depth_angle, 1e-6), 0.0, 1.0)
				confidence = clamp(0.5 + 0.5 * depth_score, 0.0, 1.0)
				self.squat_stage = "top"
				self.deepest_squat_angle = 180.0
				if self.squat_reached_depth and now - self.last_rep_times["squat"] >= SQUAT_REP_COOLDOWN:
					self.last_rep_times["squat"] = now
					self.last_motion_time = now
					return {"type": "squat", "confidence": confidence}, f"Squat: rep ({int(knee_angle)} deg)"
				self.squat_reached_depth = False

		depth_text = "depth" if self.squat_reached_depth else "no depth"
		return None, f"Squat: {self.squat_stage}, {depth_text} ({int(knee_angle)} deg)"

	def _detect_pushup(self, landmarks):
		side = self._active_side(landmarks)
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

		left_elbow_angle = self._ema("pushup_left_elbow", left_elbow_angle)
		right_elbow_angle = self._ema("pushup_right_elbow", right_elbow_angle)
		descent_angle = min(left_elbow_angle, right_elbow_angle)
		ascent_angle = average([left_elbow_angle, right_elbow_angle])
		body_is_horizontal, shoulder_hip_gap, hip_ankle_gap, body_line_angle = self._body_alignment(landmarks, side)

		now = time.perf_counter()
		if body_is_horizontal and self.pushup_stage == "top" and descent_angle < self.pushup_down_angle:
			self.pushup_stage = "descending"
			self.pushup_reached_depth = False
			self.deepest_pushup_angle = descent_angle
			self.last_motion_time = now
		elif self.pushup_stage == "descending":
			self.deepest_pushup_angle = min(self.deepest_pushup_angle, descent_angle)
			if descent_angle < self.pushup_depth_angle and body_is_horizontal:
				self.pushup_reached_depth = True
				self.last_motion_time = now
			if ascent_angle > self.pushup_up_angle and body_is_horizontal:
				bend_score = clamp((self.pushup_down_angle - self.deepest_pushup_angle) / max(self.pushup_down_angle - self.pushup_depth_angle, 1e-6), 0.0, 1.0)
				alignment_score = clamp(
					1.0 - ((shoulder_hip_gap / 0.24) + (hip_ankle_gap / 0.30)) / 2,
					0.0,
					1.0,
				)
				line_score = clamp((body_line_angle - self.body_line_min_angle) / 16.0, 0.0, 1.0)
				confidence = clamp(0.55 * bend_score + 0.25 * alignment_score + 0.20 * line_score, 0.0, 1.0)
				self.pushup_stage = "top"
				self.deepest_pushup_angle = 180.0
				if self.pushup_reached_depth and now - self.last_rep_times["pushup"] >= PUSHUP_REP_COOLDOWN:
					self.last_rep_times["pushup"] = now
					self.last_motion_time = now
					return {"type": "pushup", "confidence": confidence}, f"Push-up: rep ({int(ascent_angle)} deg)"
				self.pushup_reached_depth = False
			if not body_is_horizontal and ascent_angle > self.pushup_up_angle:
				self.pushup_stage = "top"
				self.deepest_pushup_angle = 180.0
				self.pushup_reached_depth = False

		mode_text = "ready" if body_is_horizontal else "not ready"
		depth_text = "depth" if self.pushup_reached_depth else "no depth"
		return None, f"Push-up: {self.pushup_stage}, {mode_text}, {depth_text} (min {int(descent_angle)} avg {int(ascent_angle)} deg)"