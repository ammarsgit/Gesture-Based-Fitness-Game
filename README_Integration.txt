System Integration + Game Logic (Manar)

This part of the project contains the basic foundation needed for the full webcam-based exercise detection system. 
These files now include the first full connection from live pose tracking to exercise events and game actions.

Files Included:
- game_logic.py
- mock_events.py
- poseTracker.py
- exercise_detection.py

Note:
- requirements.txt was created by Ammar and already exists in the repository.

What Each File Does:

1. game_logic.py
   Handles the core game logic:
   - Receives exercise events (squat/push-up)
   - Applies cooldowns to prevent spam
   - Filters low-confidence detections
   - Maps exercises to game actions:
       squat  -> jump
       pushup -> boost
   This file will be called by the pose detection module once exercise detection is added.

2. mock_events.py
   A temporary testing script that simulates exercise events without the webcam.
   This is only for testing the logic layer until the real detection is connected.
   Run with: python mock_events.py

3. poseTracker.py
   Working pose tracking script using MediaPipe.
   This script now shows live pose tracking with the webcam and sends detected rep events into the game logic.
   It calls:
       handle_event({"type": "squat", "confidence": value})
       handle_event({"type": "pushup", "confidence": value})
   That means webcam -> detection -> game action is now connected in the main tracker.

4. exercise_detection.py
   Simple rep-detection module.
   It uses pose landmarks to estimate:
   - squat reps from knee angle changes
   - push-up reps from elbow angle changes and a basic horizontal-body check
   It sends event dictionaries that match the format expected by game_logic.py.

Current Status:
- Webcam pose tracking works
- Game logic foundation is complete
- Cooldowns and anti-spam logic are implemented
- Mock testing system works
- Exercise detection is connected in the main tracker
- Live webcam detection can now trigger jump and boost actions through handle_event(...)

Notes:
- The current detector is intentionally simple and threshold-based so the team has a working end-to-end baseline.
- Thresholds will likely need tuning with real camera testing for different users and camera angles.