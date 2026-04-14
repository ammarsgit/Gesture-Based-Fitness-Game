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
            {"name": "Overhead Reach", "duration": 12},
            {"name": "Left Side Raise", "duration": 12},
            {"name": "Right Side Raise", "duration": 12},
            {"name": "T Pose Hold", "duration": 12},
            {"name": "Shoulder Rolls", "duration": 12},
            {"name": "Arm Circles", "duration": 12},
            {"name": "Wrist Circles", "duration": 12},
            {"name": "Rest", "duration": 8},
        ]

        self.current_step_index = 0
        self.step_start_time = None
        self.completed = False

        # setup, transition, ready_check, exercise, complete
        self.mode = "setup"
        self.start_requested = False

        self.transition_start_time = None
        self.transition_duration = 3

        self.current_feedback = "Step into frame"
        self.current_feedback_kind = "instruction"
        self.last_debug_text = "Setup"
        self.status_text = "Waiting"

        self.remaining_seconds = 0
        self.progress = 0.0

        # Slower, calmer feedback
        self.feedback_delay = 1.8
        self.pending_correction = None
        self.pending_since = None

        # Keep good feedback on screen longer
        self.last_good_time = 0.0
        self.good_hold_time = 1.5

        self.previous_landmarks = None

    def request_start(self):
        self.start_requested = True

    def restart(self):
        self.__init__()

    def get_current_step_name(self):
        if self.completed:
            return "Complete"
        if self.mode == "setup":
            return "Setup"
        if self.mode == "transition":
            return "Get Ready"
        return self.steps[self.current_step_index]["name"]

    def get_step_count_text(self):
        if self.completed:
            return "Done"
        if self.mode == "setup":
            return "Setup"
        if self.mode == "transition":
            return "Up Next"
        return f"Step {self.current_step_index + 1}/{len(self.steps)}"

    def update(self, landmarks):
        now = time.time()

        if self.completed:
            self.current_feedback = "Workout complete - press T to restart"
            self.current_feedback_kind = "good"
            self.last_debug_text = "Finished all exercises"
            self.status_text = "Complete"
            self.remaining_seconds = 0
            self.progress = 1.0
            self.previous_landmarks = landmarks
            return []

        if self.mode == "setup":
            self.previous_landmarks = landmarks
            return self._handle_setup(landmarks)

        if self.mode == "transition":
            self.previous_landmarks = landmarks
            return self._handle_transition(now)

        if self.mode == "ready_check":
            self.previous_landmarks = landmarks
            return self._handle_ready_check(landmarks)

        if self.mode == "exercise":
            events = self._handle_exercise(landmarks, now)
            self.previous_landmarks = landmarks
            return events

        self.previous_landmarks = landmarks
        return []

    def _handle_setup(self, landmarks):
        self.progress = 0.0
        self.remaining_seconds = 0

        if landmarks is None:
            self.current_feedback = "Step into frame"
            self.current_feedback_kind = "instruction"
            self.last_debug_text = "Camera cannot see you"
            self.status_text = "Not detected"
            return []

        if not self._is_centered(landmarks):
            self.current_feedback = "Move to center"
            self.current_feedback_kind = "instruction"
            self.last_debug_text = "Center your shoulders in the camera"
            self.status_text = "Adjust position"
            return []

        if not self._arms_down(landmarks):
            self.current_feedback = "Lower your arms"
            self.current_feedback_kind = "instruction"
            self.last_debug_text = "Start from a neutral pose"
            self.status_text = "Neutral pose needed"
            return []

        if not self.start_requested:
            self.current_feedback = "Press S to start"
            self.current_feedback_kind = "instruction"
            self.last_debug_text = "Ready to begin"
            self.status_text = "Ready"
            return []

        self.mode = "exercise"
        self.step_start_time = time.time()
        self.current_feedback = "Begin the movement"
        self.current_feedback_kind = "instruction"
        self.last_debug_text = self.steps[self.current_step_index]["name"]
        self.status_text = "Exercise started"
        self._clear_pending_correction()
        return [{"type": "start_session"}]

    def _handle_transition(self, now):
        elapsed = now - self.transition_start_time
        remaining = max(0, int(self.transition_duration - elapsed))
        self.remaining_seconds = remaining
        self.progress = min(1.0, max(0.0, elapsed / self.transition_duration))

        next_name = self.steps[self.current_step_index]["name"]
        self.current_feedback = f"Next: {next_name}"
        self.current_feedback_kind = "instruction"
        self.last_debug_text = "Get ready for the next exercise"
        self.status_text = "Transition"

        if elapsed >= self.transition_duration:
            self.mode = "ready_check"
            self.progress = 0.0
            self.remaining_seconds = 0

        return []

    def _handle_ready_check(self, landmarks):
        self.progress = 0.0
        self.remaining_seconds = 0

        if landmarks is None:
            self.current_feedback = "Step back into frame"
            self.current_feedback_kind = "correction"
            self.last_debug_text = "Camera lost track of you"
            self.status_text = "Not detected"
            return []

        if not self._is_centered(landmarks):
            self.current_feedback = "Move to center"
            self.current_feedback_kind = "correction"
            self.last_debug_text = "Center yourself before the next step"
            self.status_text = "Adjust position"
            return []

        if not self._arms_down(landmarks):
            self.current_feedback = "Lower your arms to begin"
            self.current_feedback_kind = "correction"
            self.last_debug_text = "Reset to neutral pose"
            self.status_text = "Neutral pose needed"
            return []

        self.mode = "exercise"
        self.step_start_time = time.time()
        self.current_feedback = f"Start: {self.steps[self.current_step_index]['name']}"
        self.current_feedback_kind = "instruction"
        self.last_debug_text = self.steps[self.current_step_index]["name"]
        self.status_text = "Ready"
        self._clear_pending_correction()
        return []

    def _handle_exercise(self, landmarks, now):
        step = self.steps[self.current_step_index]

        if self.step_start_time is None:
            self.step_start_time = now

        elapsed = now - self.step_start_time
        self.remaining_seconds = max(0, int(step["duration"] - elapsed))
        self.progress = min(1.0, max(0.0, elapsed / step["duration"]))

        if elapsed >= step["duration"]:
            self.current_step_index += 1

            if self.current_step_index >= len(self.steps):
                self.completed = True
                self.mode = "complete"
                self.current_feedback = "Workout complete - press T to restart"
                self.current_feedback_kind = "good"
                self.last_debug_text = "Finished all exercises"
                self.status_text = "Complete"
                self.remaining_seconds = 0
                self.progress = 1.0
                return [{"type": "workout_complete"}]

            self.mode = "transition"
            self.transition_start_time = now
            self._clear_pending_correction()
            return [{"type": "step_change"}]

        feedback, kind, status = self._feedback_for_current_step(step["name"], landmarks, now)
        self.current_feedback = feedback
        self.current_feedback_kind = kind
        self.status_text = status
        self.last_debug_text = step["name"]

        return []

    def _feedback_for_current_step(self, step_name, landmarks, now):
        # Hold "good" feedback a little longer so it does not flicker
        if now - self.last_good_time < self.good_hold_time:
            return "Good - keep going", "good", "Good movement"

        if landmarks is None:
            return self._delayed_correction(
                key="lost_frame",
                message="Stay in frame",
                now=now,
                default_message="Keep going",
                status="Not detected",
            )

        if not self._is_centered(landmarks):
            return self._delayed_correction(
                key="off_center",
                message="Move back to center",
                now=now,
                default_message="Keep going",
                status="Adjust position",
            )

        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]

        if step_name == "Overhead Reach":
            if left_wrist.y < left_shoulder.y and right_wrist.y < right_shoulder.y:
                self.last_good_time = now
                self._clear_pending_correction()
                return "Good - both arms are up", "good", "Good position"
            return self._delayed_correction(
                key="overhead_raise",
                message="Raise both arms higher",
                now=now,
                default_message="Reach both arms overhead",
                status="Adjust movement",
            )

        if step_name == "Left Side Raise":
            left_raised = right_wrist.y < right_shoulder.y + 0.05
            right_down = left_wrist.y > left_shoulder.y - 0.02

            if left_raised and right_down:
                self.last_good_time = now
                self._clear_pending_correction()
                return "Good - left arm is in place", "good", "Good position"
            if not right_down:
                return self._delayed_correction(
                    key="left_keep_right_down",
                    message="Keep your right arm down",
                    now=now,
                    default_message="Raise only your left arm",
                    status="Adjust movement",
                )
            return self._delayed_correction(
                key="left_raise_higher",
                message="Lift your left arm higher",
                now=now,
                default_message="Raise only your left arm",
                status="Adjust movement",
            )

        if step_name == "Right Side Raise":
            right_raised = left_wrist.y < left_shoulder.y + 0.05
            left_down = right_wrist.y > right_shoulder.y - 0.02

            if right_raised and left_down:
                self.last_good_time = now
                self._clear_pending_correction()
                return "Good - right arm is in place", "good", "Good position"
            if not left_down:
                return self._delayed_correction(
                    key="right_keep_left_down",
                    message="Keep your left arm down",
                    now=now,
                    default_message="Raise only your right arm",
                    status="Adjust movement",
                )
            return self._delayed_correction(
                key="right_raise_higher",
                message="Lift your right arm higher",
                now=now,
                default_message="Raise only your right arm",
                status="Adjust movement",
            )

        if step_name == "T Pose Hold":
            left_ok = abs(left_wrist.y - left_shoulder.y) < 0.15
            right_ok = abs(right_wrist.y - right_shoulder.y) < 0.15
            if left_ok and right_ok:
                self.last_good_time = now
                self._clear_pending_correction()
                return "Good - hold the T pose", "good", "Good position"
            return self._delayed_correction(
                key="t_pose",
                message="Stretch both arms straight out",
                now=now,
                default_message="Make a T shape with your arms",
                status="Adjust movement",
            )

        if step_name == "Shoulder Rolls":
            if self._shoulder_roll_good(landmarks):
                self.last_good_time = now
                self._clear_pending_correction()
                return "Nice - keep rolling your shoulders", "good", "Good movement"
            return self._delayed_correction(
                key="shoulder_rolls",
                message="Make smooth shoulder rolls",
                now=now,
                default_message="Roll your shoulders gently",
                status="Adjust movement",
            )

        if step_name == "Arm Circles":
            if self._arm_circles_good(landmarks):
                self.last_good_time = now
                self._clear_pending_correction()
                return "Great - keep making arm circles", "good", "Good movement"
            return self._delayed_correction(
                key="arm_circles",
                message="Make wider arm circles",
                now=now,
                default_message="Move both arms in circles",
                status="Adjust movement",
            )

        if step_name == "Wrist Circles":
            if self._wrist_circles_good(landmarks):
                self.last_good_time = now
                self._clear_pending_correction()
                return "Nice - keep circling your wrists", "good", "Good movement"
            return self._delayed_correction(
                key="wrist_circles",
                message="Rotate your wrists in small circles",
                now=now,
                default_message="Move both wrists gently",
                status="Adjust movement",
            )

        if step_name == "Rest":
            self._clear_pending_correction()
            return "Relax your arms", "instruction", "Resting"

        self._clear_pending_correction()
        return "Follow the movement", "instruction", "Instruction"

    def _delayed_correction(self, key, message, now, default_message, status):
        if self.pending_correction != key:
            self.pending_correction = key
            self.pending_since = now
            return default_message, "instruction", status

        if self.pending_since is not None and now - self.pending_since >= self.feedback_delay:
            return message, "correction", status

        return default_message, "instruction", status

    def _clear_pending_correction(self):
        self.pending_correction = None
        self.pending_since = None

    def _is_centered(self, landmarks):
        left = landmarks[poseLandmark.LEFT_SHOULDER]
        right = landmarks[poseLandmark.RIGHT_SHOULDER]
        center = (left.x + right.x) / 2.0
        shoulder_width = abs(right.x - left.x)
        return 0.38 < center < 0.62 and shoulder_width > 0.12

    def _arms_down(self, landmarks):
        left_wrist = landmarks[poseLandmark.LEFT_WRIST]
        right_wrist = landmarks[poseLandmark.RIGHT_WRIST]
        left_shoulder = landmarks[poseLandmark.LEFT_SHOULDER]
        right_shoulder = landmarks[poseLandmark.RIGHT_SHOULDER]
        return left_wrist.y > left_shoulder.y - 0.02 and right_wrist.y > right_shoulder.y - 0.02

    def _is_highest_landmark(self, landmarks, landmark_index, min_visibility=0.35, tolerance=0.015):
        target = landmarks[landmark_index]
        target_visibility = getattr(target, "visibility", 1.0)

        if target_visibility < min_visibility:
            return False

        visible_y_values = [
            lm.y
            for lm in landmarks
            if getattr(lm, "visibility", 1.0) >= min_visibility
        ]

        if not visible_y_values:
            visible_y_values = [lm.y for lm in landmarks]

        highest_y = min(visible_y_values)
        return target.y <= highest_y + tolerance

    def _shoulder_roll_good(self, landmarks):
        if self.previous_landmarks is None:
            return False

        ls = landmarks[poseLandmark.LEFT_SHOULDER]
        rs = landmarks[poseLandmark.RIGHT_SHOULDER]
        pls = self.previous_landmarks[poseLandmark.LEFT_SHOULDER]
        prs = self.previous_landmarks[poseLandmark.RIGHT_SHOULDER]

        vertical = abs(ls.y - pls.y) + abs(rs.y - prs.y)

        lw = landmarks[poseLandmark.LEFT_WRIST]
        rw = landmarks[poseLandmark.RIGHT_WRIST]
        plw = self.previous_landmarks[poseLandmark.LEFT_WRIST]
        prw = self.previous_landmarks[poseLandmark.RIGHT_WRIST]

        arm_motion = abs(lw.x - plw.x) + abs(rw.x - prw.x)

        return vertical > 0.006 and arm_motion < 0.01

    def _arm_circles_good(self, landmarks):
        if self.previous_landmarks is None:
            return False

        lw = landmarks[poseLandmark.LEFT_WRIST]
        rw = landmarks[poseLandmark.RIGHT_WRIST]
        plw = self.previous_landmarks[poseLandmark.LEFT_WRIST]
        prw = self.previous_landmarks[poseLandmark.RIGHT_WRIST]

        left_motion = abs(lw.x - plw.x) + abs(lw.y - plw.y)
        right_motion = abs(rw.x - prw.x) + abs(rw.y - prw.y)
        total_motion = (left_motion + right_motion) / 2.0

        le = landmarks[poseLandmark.LEFT_ELBOW]
        re = landmarks[poseLandmark.RIGHT_ELBOW]
        ls = landmarks[poseLandmark.LEFT_SHOULDER]
        rs = landmarks[poseLandmark.RIGHT_SHOULDER]

        elbows_extended = (
            abs(le.y - ls.y) < 0.2 and
            abs(re.y - rs.y) < 0.2
        )

        return total_motion > 0.012 and elbows_extended

    def _wrist_circles_good(self, landmarks):
        if self.previous_landmarks is None:
            return False

        lw = landmarks[poseLandmark.LEFT_WRIST]
        rw = landmarks[poseLandmark.RIGHT_WRIST]
        plw = self.previous_landmarks[poseLandmark.LEFT_WRIST]
        prw = self.previous_landmarks[poseLandmark.RIGHT_WRIST]

        wrist_motion = (
            abs(lw.x - plw.x) + abs(lw.y - plw.y) +
            abs(rw.x - prw.x) + abs(rw.y - prw.y)
        ) / 2.0

        le = landmarks[poseLandmark.LEFT_ELBOW]
        re = landmarks[poseLandmark.RIGHT_ELBOW]
        ple = self.previous_landmarks[poseLandmark.LEFT_ELBOW]
        pre = self.previous_landmarks[poseLandmark.RIGHT_ELBOW]

        elbow_motion = abs(le.x - ple.x) + abs(re.x - pre.x)

        return wrist_motion > 0.006 and elbow_motion < 0.01