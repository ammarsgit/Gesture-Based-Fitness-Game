import time
from game_logic import handle_event

# Simulated sequence of detected exercises
mock_events = [
    {"type": "squat", "confidence": 0.95},
    {"type": "squat", "confidence": 0.95},  # should be blocked by cooldown
    {"type": "pushup", "confidence": 0.90},
    {"type": "pushup", "confidence": 0.60},  # low confidence, ignored
    {"type": "pushup", "confidence": 0.92},  # may be blocked or allowed depending on timing
]

for event in mock_events:
    print(f"\nEVENT: {event}")
    handle_event(event)
    time.sleep(0.3)  # 300ms between events