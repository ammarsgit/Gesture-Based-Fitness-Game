import time

from game_logic import handle_event


mock_events = [
	{"type": "squat", "confidence": 0.95},
	{"type": "squat", "confidence": 0.95},
	{"type": "pushup", "confidence": 0.90},
	{"type": "pushup", "confidence": 0.60},
	{"type": "pushup", "confidence": 0.92},
]

for event in mock_events:
	print(f"\nEVENT: {event}")
	handle_event(event)
	time.sleep(0.3)