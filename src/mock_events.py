import time
from game_logic import handle_event

mock_events = [
    {"type": "not_in_frame", "confidence": 1.0},
    {"type": "start_session", "confidence": 1.0},
    {"type": "step_change", "confidence": 1.0},
    {"type": "workout_complete", "confidence": 1.0},
]

for event in mock_events:
    print(f"\nEVENT SENT: {event}")
    handle_event(event)
    time.sleep(0.6)