System Integration + Game Logic (Manar)

This part of the project contains the basic foundation needed for the full webcam-based exercise detection system. 
These files are not the final gameplay system yet, they are the required components that will connect Ammar’s pose detection to the game actions once his exercise detection code is added.

Files Included:
- game_logic.py
- mock_events.py
- poseTracker.py

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
   This script currently shows live pose tracking with the webcam.
   Once Ammar uploads his exercise detection code, it will call:
       handle_event({"type": "squat", "confidence": value})
       handle_event({"type": "pushup", "confidence": value})
   This will allow full end-to-end testing with the webcam.

Current Status:
- Webcam pose tracking works
- Game logic foundation is complete
- Cooldowns and anti-spam logic are implemented
- Mock testing system works
- Waiting for Ammar’s exercise detection module to enable full webcam-based gameplay