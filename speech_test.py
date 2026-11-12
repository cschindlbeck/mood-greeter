"""Standalone test of the lynx voice: renders and plays a few lines in order."""
from lynx_voice import Voice

if __name__ == "__main__":
    print("Testing speech output...")

    voice = Voice(publish=lambda event: print({**event, "envelope": "..."} if event.get("envelope") else event))
    voice.speak("This is a speech test.")
    voice.speak("Welcome to the LYNQTECH Christmas party!")
    voice.speak("This is the last line.")

    # Wait for all speech to finish
    voice.wait()
    voice.close()
    print("Speech test complete!")
