def handle_event(event):
    event_type = event.get("type")

    if event_type == "start_session":
        print("Workout started")
    elif event_type == "step_change":
        print("Next exercise")
    elif event_type == "workout_complete":
        print("Workout complete")