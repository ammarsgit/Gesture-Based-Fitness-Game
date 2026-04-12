import time

last_event_times = {}

cooldowns = {
    "not_in_frame": 1.5,
    "start_session": 1.0,
    "step_change": 1.0,
    "workout_complete": 2.0,
}


def handle_event(event):
    event_type = event.get("type", "unknown")
    now = time.time()

    last_time = last_event_times.get(event_type, 0.0)
    cooldown = cooldowns.get(event_type, 0.5)
    if now - last_time < cooldown:
        return

    last_event_times[event_type] = now

    if event_type == "not_in_frame":
        print("EVENT: User not in frame")
    elif event_type == "start_session":
        print("EVENT: Session started")
    elif event_type == "step_change":
        print("EVENT: Step changed")
    elif event_type == "workout_complete":
        print("EVENT: Workout complete")
    else:
        print(f"EVENT: {event_type}")