import time


last_squat_time = 0
last_pushup_time = 0
last_plank_time = 0
last_shoulder_tap_time = 0

SQUAT_COOLDOWN = 0.8
PUSHUP_COOLDOWN = 1.0
PLANK_COOLDOWN = 2.0
SHOULDER_TAP_COOLDOWN = 0.4


def handle_event(event):
	global last_squat_time, last_pushup_time, last_plank_time, last_shoulder_tap_time
	now = time.time()

	if event.get("confidence", 0) < 0.7:
		print(f"IGNORED low confidence {event['type']} ({event['confidence']:.2f})")
		return

	if event["type"] == "squat":
		if now - last_squat_time > SQUAT_COOLDOWN:
			print("GAME ACTION: JUMP")
			last_squat_time = now
		else:
			print("IGNORED squat (cooldown)")

	elif event["type"] == "pushup":
		if now - last_pushup_time > PUSHUP_COOLDOWN:
			print("GAME ACTION: BOOST")
			last_pushup_time = now
		else:
			print("IGNORED pushup (cooldown)")

	elif event["type"] == "plank":
		if now - last_plank_time > PLANK_COOLDOWN:
			print("GAME ACTION: SHIELD")
			last_plank_time = now
		else:
			print("IGNORED plank (cooldown)")

	elif event["type"] == "shoulder_tap":
		if now - last_shoulder_tap_time > SHOULDER_TAP_COOLDOWN:
			print("GAME ACTION: STRIKE")
			last_shoulder_tap_time = now
		else:
			print("IGNORED shoulder tap (cooldown)")