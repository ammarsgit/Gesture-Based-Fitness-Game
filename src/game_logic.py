import time


last_squat_time = 0
last_pushup_time = 0

SQUAT_COOLDOWN = 0.8
PUSHUP_COOLDOWN = 1.0
MIN_EVENT_CONFIDENCE = 0.6


def handle_event(event):
	global last_squat_time, last_pushup_time
	now = time.time()

	if event.get("confidence", 0) < MIN_EVENT_CONFIDENCE:
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
	else:
		print(f"IGNORED unsupported exercise {event['type']}")