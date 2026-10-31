import cv2
import time
import subprocess
import threading
import queue
from collections import Counter
from deepface import DeepFace
from deepface.commons import weight_utils

# Speech queue for non-blocking, sequential speech
speech_queue = queue.Queue()


def speech_worker():
    """Background thread that processes speech queue sequentially."""
    while True:
        text = speech_queue.get()
        if text is None:  # Shutdown signal
            break
        subprocess.run(['say', '-r', '195', text])  # Default is ~175 wpm, 195 is ~10% faster
        speech_queue.task_done()


# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def speak(text):
    """Queue text for speech. Non-blocking, but speeches play sequentially."""
    speech_queue.put(text)


def is_speaking():
    """True while speech is queued or still playing."""
    return speech_queue.unfinished_tasks > 0

# Load face cascade classifier for fast detection (much faster than DeepFace's internal detector for video)
# OpenCV 5 wheels ship no cascade files, so fetch it into DeepFace's weight cache (~/.deepface/weights)
CASCADE_FILE = 'haarcascade_frontalface_default.xml'
cascade_path = weight_utils.download_weights_if_necessary(
    file_name=CASCADE_FILE,
    source_url=f'https://github.com/opencv/opencv/raw/4.9.0/data/haarcascades/{CASCADE_FILE}',
)
face_cascade = cv2.CascadeClassifier(cascade_path)

# Sentiment detection settings
ANALYSIS_DURATION = 1.5  # Seconds to analyze before locking sentiment
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
read_failures = 0  # Consecutive failed camera reads
READ_FAILURE_THRESHOLD = 5  # Failed reads before giving up (each read times out after ~1s; some cameras need >1s to warm up)
TALLY_WIDTH = 240  # Pixels for the emotion tally next to the face

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
        read_failures += 1
        if read_failures >= READ_FAILURE_THRESHOLD:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        continue
    read_failures = 0

    # 2. Convert to grayscale for the face detector (faster)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 3. Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # 4. Hold the timers while speech plays, so the analysis window and the cooldown
    # start only after the speech has finished. One snapshot per frame keeps the
    # decisions below consistent.
    speaking = is_speaking()
    if speaking:
        if face_detected_time is not None and locked_sentiment is None:
            face_detected_time = time.time()
        if sentiment_locked_time is not None:
            sentiment_locked_time = time.time()

    # 5. Process detected faces
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
        # if new person AND previous analysis is complete.
        # Never while speech plays, so speech cannot pile up in the queue.
        if not speaking and (face_detected_time is None or (is_new_person and sentiment_announced)):
            face_detected_time = time.time()
            sentiment_samples = []
            locked_sentiment = None
            sentiment_announced = False
            # Greet the person once when analysis starts
            speak("Welcome to Lynqtech! Look at me for a moment, I'm reading your mood.")
            greeted = True
            speaking = True  # Analysis starts after the greeting

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face for DeepFace analysis
        face_roi = frame[y : y + h, x : x + w]

        if speaking:
            # No samples while speech plays, so the result matches the face after it
            if locked_sentiment is None:
                cv2.putText(frame, "SPEAKING...", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
        elif locked_sentiment is None:
            elapsed_time = time.time() - face_detected_time

            # Still in analysis period
            try:
                results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)
                result = results[0]
                emotion = result['dominant_emotion']
                sentiment_samples.append(emotion)
            except Exception:
                pass

            if elapsed_time < ANALYSIS_DURATION:
                # Show analyzing status with a progress bar under the face
                cv2.putText(frame, "READING YOUR MOOD...", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
                progress = int(w * elapsed_time / ANALYSIS_DURATION)
                cv2.rectangle(frame, (x, y + h + 10), (x + w, y + h + 25), (0, 165, 255), 2)
                cv2.rectangle(frame, (x, y + h + 10), (x + progress, y + h + 25), (0, 165, 255), -1)
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
                    elif sentiment_lower == "surprise":
                        speak("Why are you surprised? Never seen an AI Lynx before")
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

        # Show the share of each emotion so far next to the face, leader first in green
        if sentiment_samples:
            tally_x = x + w + 10
            if tally_x + TALLY_WIDTH > frame.shape[1]:
                tally_x = max(0, x - 10 - TALLY_WIDTH)  # No room on the right
            ranked = Counter(sentiment_samples).most_common()
            # Darken the area behind the tally, so the text stays readable on light backgrounds
            frame[max(0, y):y + 10 + len(ranked) * 25, max(0, tally_x - 5):tally_x + TALLY_WIDTH] //= 3
            for i, (emotion, count) in enumerate(ranked):
                share = count / len(sentiment_samples)
                row_y = y + 20 + i * 25
                color = (36, 255, 12) if i == 0 else (255, 255, 255)
                cv2.putText(frame, emotion, (tally_x, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.rectangle(frame, (tally_x + 80, row_y - 14), (tally_x + 80 + int(100 * share), row_y), color, -1)
                cv2.putText(frame, f"{share:.0%}", (tally_x + 190, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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

    # 6. Display the resulting frame
    cv2.imshow('Mac Pro Sentiment Analysis', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
