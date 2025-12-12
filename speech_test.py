import subprocess
import threading
import queue

# Speech queue for non-blocking, sequential speech
speech_queue = queue.Queue()


def speech_worker():
    """Background thread that processes speech queue sequentially."""
    while True:
        text = speech_queue.get()
        if text is None:  # Shutdown signal
            break
        subprocess.run(['say', '-r', '195', text])  # 195 wpm (~10% faster)
        speech_queue.task_done()


# Start the speech worker thread
speech_thread = threading.Thread(target=speech_worker, daemon=True)
speech_thread.start()


def speak(text):
    """Queue text for speech. Non-blocking, but speeches play sequentially."""
    speech_queue.put(text)


# Test the speech
if __name__ == "__main__":
    print("Testing speech output...")

    speak("This is a speech test")
    speak("This is a speech test")
    speak("This is a speech test")
    speak("This is a speech test")

    # Wait for all speech to finish
    speech_queue.join()
    print("Speech test complete!")
