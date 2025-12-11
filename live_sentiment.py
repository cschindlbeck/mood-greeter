import cv2
import time
import subprocess
import threading
import queue
from collections import Counter
from deepface import DeepFace

# Speech queue for non-blocking, sequential speech
speech_queue = queue.Queue()


def speech_worker():
    """Background thread that processes speech queue sequentially."""
    while True:
        text = speech_queue.get()
        if text is None:  # Shutdown signal
            break
        subprocess.run(['say', text])
        speech_queue.task_done()


# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def speak(text):
    """Queue text for speech. Non-blocking, but speeches play sequentially."""
    speech_queue.put(text)

# Load face cascade classifier for fast detection (much faster than DeepFace's internal detector for video)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Sentiment detection settings
ANALYSIS_DURATION = 3  # Seconds to analyze before locking sentiment
sentiment_samples = []  # Collect samples during analysis period
locked_sentiment = None  # The final locked sentiment
face_detected_time = None  # When face was first detected
no_face_frames = 0  # Counter for frames without a face
NO_FACE_THRESHOLD = 15  # Frames without face before resetting
last_face_position = None  # Track face position to detect new person
POSITION_CHANGE_THRESHOLD = 150  # Pixels - if face moves more than this, it's a new person
greeted = False  # Track if we've greeted the current person
sentiment_announced = False  # Track if we've announced the sentiment
sentiment_locked_time = None  # When sentiment was locked (for cooldown)
COOLDOWN_DURATION = 5  # Seconds to wait before new analysis

# Initialize webcam (0 is usually the default Mac webcam)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

print("--- Press 'q' to quit ---")

while True:
    # 1. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # 2. Convert to grayscale for the face detector (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 4. Process detected faces
    if len(faces) > 0:
        no_face_frames = 0  # Reset no-face counter

        # Find the largest face (closest to camera / in front)
        largest_face = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = largest_face
        current_face_center = (x + w // 2, y + h // 2)

        # Check if this is a new person (significant position change)
        is_new_person = False
        if last_face_position is not None:
            dx = abs(current_face_center[0] - last_face_position[0])
            dy = abs(current_face_center[1] - last_face_position[1])
            if dx > POSITION_CHANGE_THRESHOLD or dy > POSITION_CHANGE_THRESHOLD:
                is_new_person = True

        last_face_position = current_face_center

        # Start timing when face first detected or new person appears
        # Only restart if no active analysis (face_detected_time is None) or
        # if new person AND previous analysis is complete
        if face_detected_time is None or (is_new_person and sentiment_announced):
            face_detected_time = time.time()
            sentiment_samples = []
            locked_sentiment = None
            sentiment_announced = False
            # Greet the person once when analysis starts
            speak("Welcome to Lynqtech, how are you today?")
            greeted = True

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face for DeepFace analysis
        face_roi = frame[y : y + h, x : x + w]

        elapsed_time = time.time() - face_detected_time

        if locked_sentiment is None:
            # Still in analysis period
            try:
                results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                result = results[0]
                emotion = result['dominant_emotion']
                sentiment_samples.append(emotion)
            except Exception:
                pass

            if elapsed_time < ANALYSIS_DURATION:
                # Show analyzing status with countdown
                remaining = ANALYSIS_DURATION - elapsed_time
                cv2.putText(frame, f"ANALYZING... {remaining:.1f}s", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
            else:
                # Lock the dominant sentiment
                if sentiment_samples:
                    locked_sentiment = Counter(sentiment_samples).most_common(1)[0][0]
                else:
                    locked_sentiment = "neutral"

                # Announce the sentiment
                if not sentiment_announced:
                    speak(f"You seem to be feeling {locked_sentiment}")
                    sentiment_lower = locked_sentiment.lower()
                    if sentiment_lower == "happy":
                        speak("You are here to work, not to be happy")
                    elif sentiment_lower == "sad":
                        speak("Don't be sad, think about your next paycheck")
                    elif sentiment_lower == "angry":
                        speak("Don't be angry, the weekend is around the corner")
                    elif sentiment_lower == "fear":
                        speak("Don't be afraid, all people are nice here, except for Mark")
                    elif sentiment_lower == "neutral":
                        speak("Hey, please be happy, you work at lynqtech")
                    else:
                        speak("Unknown emotions: You are not a robot, or are you?")
                    sentiment_announced = True
                    sentiment_locked_time = time.time()

        # Check if cooldown period is over, then reset for new analysis
        if sentiment_announced and sentiment_locked_time:
            if time.time() - sentiment_locked_time > COOLDOWN_DURATION:
                face_detected_time = None
                locked_sentiment = None
                sentiment_samples = []
                last_face_position = None
                greeted = False
                sentiment_announced = False
                sentiment_locked_time = None

        if locked_sentiment:
            # Display the locked sentiment
            cv2.putText(frame, locked_sentiment.upper(), (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    else:
        # No face detected
        no_face_frames += 1
        if no_face_frames > NO_FACE_THRESHOLD:
            # Reset for next person
            face_detected_time = None
            locked_sentiment = None
            sentiment_samples = []
            last_face_position = None
            greeted = False
            sentiment_announced = False
            sentiment_locked_time = None

    # 5. Display the resulting frame
    cv2.imshow('Mac Pro Sentiment Analysis', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
