import time
from mediapipe.tasks.python import vision

poseLandmark = vision.PoseLandmark


def point_distance(a, b):
    dx = a.x - b.x
    dy = a.y - b.y
    return (dx * dx + dy * dy) ** 0.5


class ExerciseDetector:
    def __init__(self):
        self.steps = [
            {"name": "Overhead Reach", "duration": 15},
            {"name": "Left Side Raise", "duration": 15},
            {"name": "Right Side Raise", "duration": 15},
            {"name": "T Pose Hold", "duration": 15},
            {"name": "Shoulder Rolls", "duration": 15},   # NEW
            {"name": "Arm Circles", "duration": 15},       # NEW
            {"name": "Rest", "duration": 10},
        ]

        self.current_step_index = 0
        self.step_start_time = None
        self.previous_landmarks = None
        self.completed = False

        self.ready_state = "waiting_for_frame"
        self.start_requested = False

        self.last_debug_text = "Step into the camera view"
        self.current_feedback = "Waiting for user"
        self.remaining_seconds = 0
        self.progress = 0.0

        self.feedback_update_interval = 0.35
        self.last_feedback_update_time = 0.0
        self.cached_feedback = ""

    def request_start(self):
        self.start_requested = True

    def restart(self):
        self.current_step_index = 0
        self.step_start_time = None
        self.previous_landmarks = None
        self.completed = False
        self.ready_state = "waiting_for_frame"
        self.start_requested = False
        self.last_debug_text = "Step into the camera view"
        self.current_feedback = "Waiting for user"
        self.remaining_seconds = 0
        self.progress = 0.0
        self.last_feedback_update_time = 0.0
        self.cached_feedback = ""

    def get_current_step_name(self):
        if self.completed:
            return "Complete"
        if self.ready_state != "ready":
            return "Setup"
        return self.steps[self.current_step_index]["name"]

    def get_step_count_text(self):
        if self.ready_state != "ready":
            return "Setup"
        return f"Step {self.current_step_index + 1}/{len(self.steps)}"

    def update(self, landmarks):
        now = time.time()

        if self.completed:
            self.last_debug_text = "Workout complete - press T to restart"
            self.current_feedback = "Great job"
            self.remaining_seconds = 0
            self.progress = 1.0
            return []

        if self.ready_state != "ready":
            return self._handle_ready_state(landmarks)

        return self._run_workout(landmarks, now)

    def _handle_ready_state(self, landmarks):
        self.progress = 0.0
        self.remaining_seconds = 0

        if landmarks is None:
            self.ready_state = "waiting_for_frame"
            self.last_debug_text = "Step into the camera view"
            self.current_feedback = "Make sure your shoulders are visible"
            self.previous_landmarks = None
            return [{"type": "not_in_frame", "confidence": 1.0}]

        if not self._is_centered(landmarks):
            self.ready_state = "move_to_center"
            self.last_debug_text = "Move to the center"
            self.current_feedback = "Try to line up your shoulders with the middle"
            self.previous_landmarks = landmarks
            return []

        if not self.start_requested:
            self.ready_state = "ready_to_start"
            self.last_debug_text = "Press S to start"
            self.current_feedback = "You are centered and ready"
            self.previous_landmarks = landmarks
            return []

        self.ready_state = "ready"
        self.step_start_time = time.time()
        self.start_requested = False
        self.previous_landmarks = landmarks
        self.last_feedback_update_time = 0.0
        self.cached_feedback = ""
        self.last_debug_text = f"Starting: {self.steps[self.current_step_index]['name']}"
        self.current_feedback = "Follow the movement on screen"
        return [{"type": "start_session", "confidence": 1.0}]

    def _run_workout(self, landmarks, now):
        if landmarks is None:
            self.ready_state = "waiting_for_frame"
            self.step_start_time = None
            self.start_requested = False
            self.previous_landmarks = None
            self.cached_feedback = ""
            self.last_debug_text = "Person lost - step back into frame"
            self.current_feedback = "The workout is paused"
            self.remaining_seconds = 0
            self.progress = 0.0
            return [{"type": "not_in_frame", "confidence": 1.0}]

        if self.step_start_time is None:
            self.step_start_time = now

        step = self.steps[self.current_step_index]
        duration = step["duration"]
        elapsed = now - self.step_start_time
        remaining = max(0, int(duration - elapsed))
        self.remaining_seconds = remaining
        self.progress = min(1.0, max(0.0, elapsed / duration))

        if elapsed >= duration:
            finished_step = step["name"]
            self.current_step_index += 1
            self.step_start_time = now
            self.previous_landmarks = landmarks
            self.cached_feedback = ""
            self.last_feedback_update_time = 0.0

            if self.current_step_index >= len(self.steps):
                self.completed = True
                self.last_debug_text = "Workout complete - press T to restart"
                self.current_feedback = "Great job"
                self.remaining_seconds = 0
                self.progress = 1.0
                return [{"type": "workout_complete", "confidence": 1.0}]

            next_step = self.steps[self.current_step_index]["name"]
            self.last_debug_text = f"Finished: {finished_step}"
            self.current_feedback = f"Next: {next_step}"
            self.remaining_seconds = self.steps[self.current_step_index]["duration"]
            self.progress = 0.0
            return [{"type": "step_change", "confidence": 1.0}]

        movement_level = self._movement_level(landmarks)

        if now - self.last_feedback_update_time >= self.feedback_update_interval:
            self.cached_feedback = self._feedback_for_step(step["name"], landmarks, movement_level)
            self.last_feedback_update_time = now

        self.last_debug_text = f"{step['name']} - {remaining}s left"
        self.current_feedback = self.cached_feedback

        self.previous_landmarks = landmarks
        return []

    def _is_centered(self, landmarks):
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
        center_x = (left_shoulder.x + right_shoulder.x) / 2.0
        shoulder_width = abs(right_shoulder.x - left_shoulder.x)
        return 0.38 < center_x < 0.62 and shoulder_width > 0.12

    def _movement_level(self, landmarks):
        if self.previous_landmarks is None:
            return 0.0

        tracked_points = [
            poseLandmark.LEFT_WRIST,
            poseLandmark.RIGHT_WRIST,
            poseLandmark.LEFT_ELBOW,
            poseLandmark.RIGHT_ELBOW,
            poseLandmark.LEFT_SHOULDER,
            poseLandmark.RIGHT_SHOULDER,
        ]

        total_motion = 0.0
        for idx in tracked_points:
            total_motion += point_distance(landmarks[idx], self.previous_landmarks[idx])

        return total_motion / len(tracked_points)

    def _feedback_for_step(self, step_name, landmarks, movement_level):

        if step_name == "Overhead Reach":
            left_up, right_up = self._arms_overhead_flags(landmarks)
            if left_up and right_up:
                return "Great - both arms are up"
            if left_up or right_up:
                return "Raise both arms together"
            if movement_level > 0.012:
                return "Lift your hands higher"
            return "Reach both arms overhead"

        if step_name == "Left Side Raise":
            left_good = self._left_side_good(landmarks)
            right_up = self._right_arm_not_down(landmarks)
            left_partial = self._left_arm_partial(landmarks)
            if left_good:
                return "Great - left arm is in position"
            if right_up:
                return "Keep your right arm down"
            if left_partial:
                return "Lift your left arm a little higher"
            if movement_level > 0.012:
                return "Move only your left arm outward"
            return "Raise your left arm to shoulder height"

        if step_name == "Right Side Raise":
            right_good = self._right_side_good(landmarks)
            left_up = self._left_arm_not_down(landmarks)
            right_partial = self._right_arm_partial(landmarks)
            if right_good:
                return "Great - right arm is in position"
            if left_up:
                return "Keep your left arm down"
            if right_partial:
                return "Lift your right arm a little higher"
            if movement_level > 0.012:
                return "Move only your right arm outward"
            return "Raise your right arm to shoulder height"

        if step_name == "T Pose Hold":
            both_good = self._t_pose_good(landmarks)
            left_partial = self._left_arm_partial(landmarks)
            right_partial = self._right_arm_partial(landmarks)
            if both_good:
                return "Great - hold both arms straight out"
            if left_partial or right_partial:
                return "Lift both arms to shoulder height"
            if movement_level > 0.012:
                return "Stretch both arms outward"
            return "Make a T shape with your arms"

        if step_name == "Shoulder Rolls":
            good_roll = self._shoulder_roll_good(landmarks, movement_level)
            if good_roll:
                return "Nice - keep rolling your shoulders"
            if movement_level > 0.010:
                return "Make smooth circles with your shoulders"
            return "Gently roll your shoulders forward and back"

        if step_name == "Arm Circles":
            good_circle = self._arm_circles_good(landmarks, movement_level)
            if good_circle:
                return "Great – keep making smooth circles"
            if movement_level > 0.010:
                return "Make wider circles with your arms"
            return "Rotate your arms in a circular motion"

        if step_name == "Rest":
            if movement_level < 0.008:
                return "Nice - stay relaxed"
            return "Relax your arms and stay still"

        return "Follow the on-screen instruction"

    def _arms_overhead_flags(self, landmarks):
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
        left_up = left_wrist.y < left_shoulder.y - 0.04
        right_up = right_wrist.y < right_shoulder.y - 0.04
        return left_up, right_up

    def _left_arm_partial(self, landmarks):
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        return left_wrist.x < left_shoulder.x - 0.08 and abs(left_wrist.y - left_shoulder.y) < 0.22

    def _right_arm_partial(self, landmarks):
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
        return right_wrist.x > right_shoulder.x + 0.08 and abs(right_wrist.y - right_shoulder.y) < 0.22

    def _left_arm_not_down(self, landmarks):
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        return left_wrist.y < left_shoulder.y + 0.02

    def _right_arm_not_down(self, landmarks):
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
        return right_wrist.y < right_shoulder.y + 0.02

    def _left_side_good(self, landmarks):
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        left_elbow = landmarks[poseLandmark.LEFT_ELBOW]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]

        left_at_height = abs(left_wrist.y - left_shoulder.y) < 0.14
        left_outward = left_wrist.x < left_shoulder.x - 0.12
        left_elbow_out = left_elbow.x < left_shoulder.x - 0.05
        right_down = right_wrist.y > right_shoulder.y + 0.03

        return left_at_height and left_outward and left_elbow_out and right_down

    def _right_side_good(self, landmarks):
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        right_elbow = landmarks[poseLandmark.RIGHT_ELBOW]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]

        right_at_height = abs(right_wrist.y - right_shoulder.y) < 0.14
        right_outward = right_wrist.x > right_shoulder.x + 0.12
        right_elbow_out = right_elbow.x > right_shoulder.x + 0.05
        left_down = left_wrist.y > left_shoulder.y + 0.03

        return right_at_height and right_outward and right_elbow_out and left_down

    def _t_pose_good(self, landmarks):
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[poseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[poseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]

        left_good = (
            abs(left_wrist.y - left_shoulder.y) < 0.14
            and left_wrist.x < left_shoulder.x - 0.12
            and left_elbow.x < left_shoulder.x - 0.05
        )
        right_good = (
            abs(right_wrist.y - right_shoulder.y) < 0.14
            and right_wrist.x > right_shoulder.x + 0.12
            and right_elbow.x > right_shoulder.x + 0.05
        )
        return left_good and right_good

    def _shoulder_roll_good(self, landmarks, movement_level):
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
        left_elbow = landmarks[poseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[poseLandmark.RIGHT_ELBOW]

        if self.previous_landmarks is None:
            return False

        prev_left_shoulder = self.previous_landmarks[poseLandmark.LEFT_SHOULDER]
        prev_right_shoulder = self.previous_landmarks[poseLandmark.RIGHT_SHOULDER]

        left_dy = abs(left_shoulder.y - prev_left_shoulder.y)
        right_dy = abs(right_shoulder.y - prev_right_shoulder.y)
        avg_shoulder_vertical_motion = (left_dy + right_dy) / 2.0

        left_elbow_close = abs(left_elbow.x - left_shoulder.x) < 0.10
        right_elbow_close = abs(right_elbow.x - right_shoulder.x) < 0.10

        enough_motion = movement_level > 0.006
        enough_shoulder_vertical = avg_shoulder_vertical_motion > 0.004

        return enough_motion and enough_shoulder_vertical and left_elbow_close and right_elbow_close

    def _arm_circles_good(self, landmarks, movement_level):
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        left_elbow = landmarks[poseLandmark.LEFT_ELBOW]
        right_elbow = landmarks[poseLandmark.RIGHT_ELBOW]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]

        if self.previous_landmarks is None:
            return False

        prev_left_wrist = self.previous_landmarks[poseLandmark.LEFT_WRIST]
        prev_right_wrist = self.previous_landmarks[poseLandmark.RIGHT_WRIST]

        left_vertical = abs(left_wrist.y - prev_left_wrist.y)
        right_vertical = abs(right_wrist.y - prev_right_wrist.y)

        left_horizontal = abs(left_wrist.x - prev_left_wrist.x)
        right_horizontal = abs(right_wrist.x - prev_right_wrist.x)

        left_circle_motion = left_vertical + left_horizontal
        right_circle_motion = right_vertical + right_horizontal

        enough_motion = movement_level > 0.008
        enough_circle = (left_circle_motion + right_circle_motion) / 2.0 > 0.010

        left_elbow_extended = abs(left_elbow.y - left_shoulder.y) < 0.20
        right_elbow_extended = abs(right_elbow.y - right_shoulder.y) < 0.20

        return enough_motion and enough_circle and left_elbow_extended and right_elbow_extended
