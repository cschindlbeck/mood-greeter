import cv2
from deepface import DeepFace

# Load face cascade classifier for fast detection (much faster than DeepFace's internal detector for video)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    # 4. Process each detected face
    for x, y, w, h in faces:
        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Crop the face for DeepFace analysis
        face_roi = frame[y : y + h, x : x + w]

        try:
            # Analyze sentiment
            # enforce_detection=False prevents crash if face is blurry
            results = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False, silent=True)

            # DeepFace returns a list, grab the first result
            result = results[0]
            emotion = result['dominant_emotion']

            # Put text on screen
            cv2.putText(frame, emotion.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        except Exception as e:
            pass  # Skip frame if analysis fails

    # 5. Display the resulting frame
    cv2.imshow('Mac Pro Sentiment Analysis', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
