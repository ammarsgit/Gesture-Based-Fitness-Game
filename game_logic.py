import time

# Track last time each action was triggered
last_squat_time = 0
last_pushup_time = 0

# Cooldowns in seconds
SQUAT_COOLDOWN = 0.8
PUSHUP_COOLDOWN = 1.0

def handle_event(event):
    """
    event: dict with at least:
        - "type": "squat" or "pushup"
        - "confidence": float between 0 and 1
    """
    global last_squat_time, last_pushup_time
    now = time.time()

    # Ignore low-confidence events
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
            