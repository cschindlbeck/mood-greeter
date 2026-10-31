import time

from deepface import DeepFace


def analyze_sentiment(image_path):
    print(f"--- Analyzing: {image_path} ---")

    start_time = time.time()

    try:
        # DeepFace.analyze does the heavy lifting: detection + sentiment
        # actions=['emotion'] restricts it to just sentiment (faster)
        objs = DeepFace.analyze(
            img_path=image_path,
            actions=['emotion'],
            enforce_detection=False,  # Set to True to crash if no face found
            detector_backend='opencv',  # 'opencv' is fastest; use 'retinaface' for accuracy
        )

        # DeepFace returns a list (in case of multiple faces)
        # We'll take the first face found for this example
        result = objs[0]

        emotion_profile = result['emotion']
        dominant_emotion = result['dominant_emotion']

        elapsed = time.time() - start_time

        print(f"\n📸 Result:")
        print(f"   Dominant Emotion: {dominant_emotion.upper()}")
        print(f"   Confidence: {emotion_profile[dominant_emotion]:.2f}%")
        print(f"⏱️  Processing Time: {elapsed:.4f} seconds")

        return result

    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# --- usage ---
# Replace 'face.jpg' with the path to your image
if __name__ == "__main__":
    # Create a dummy file or use your own to test
    analyze_sentiment("angryface.jpeg")
    pass
