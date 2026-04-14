def handle_event(event):
    t = event.get("type")

    if t == "start_session":
        print("Workout started")
    elif t == "step_change":
        print("Next exercise")
    elif t == "workout_complete":
        print("Workout complete")