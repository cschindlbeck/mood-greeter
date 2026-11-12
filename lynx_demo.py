"""The lynx show without a camera: fake events, real speech.

Use it to tune the lip sync, the poses and the lines at a desk (`make demo`).
It walks through the same steps as live_sentiment.py: nobody there, greeting,
analysis, mood line, cooldown. Then the next mood.
"""
import itertools
import time

import cv2
import numpy as np

import lynx_lines
import lynx_show

# Same timings as in live_sentiment.py
ANALYSIS_DURATION = 1.5
COOLDOWN_DURATION = 5
IDLE_DURATION = 4  # Seconds with nobody there between two fake persons

parser = lynx_show.argument_parser("Lynx show without a camera, for tuning.")
parser.add_argument("--moods", default=",".join(lynx_lines.MOOD_LINES),
                    help="comma-separated moods to cycle through (default: all)")
parser.add_argument("--once", action="store_true", help="stop after one pass through the moods")
args = parser.parse_args()

show = lynx_show.Show(port=args.port, fullscreen=not args.window, browser=not args.no_browser)


def demo_frame(text, image=None):
    """A stand-in for the camera frame."""
    frame = np.full((720, 1280, 3), 40, np.uint8)
    if image is not None:
        height = 720
        width = image.shape[1] * height // image.shape[0]
        left = (1280 - width) // 2
        frame[:, left:left + width] = cv2.resize(image, (width, height))
    cv2.putText(frame, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 4)
    return frame


EMPTY_FRAME = demo_frame("DEMO: nobody here")
PERSON_FRAME = demo_frame("DEMO: fake person", cv2.imread(str(lynx_show.ROOT / "angryface.jpeg")))


def hold(frame, seconds=0, **state):
    """Keep a state while the lynx speaks and then for some seconds. False if the user quits."""
    end = None
    while True:
        if show.quit_requested():
            return False
        show.set_state(**state)
        show.show_frame(frame)
        if not show.is_speaking():
            end = end or time.time() + seconds  # The timers start after the speech, as in the live loop
            if time.time() >= end:
                return True
        time.sleep(1 / 30)


def greet(mood):
    """One fake person with this mood. False if the user quits."""
    if not hold(EMPTY_FRAME, IDLE_DURATION, present=False, phase="idle", mood=None):
        return False
    show.speak(show.lines.welcome())
    if not hold(PERSON_FRAME, present=True, phase="greeting", mood=None):
        return False
    if not hold(PERSON_FRAME, ANALYSIS_DURATION, present=True, phase="analysing", mood=None):
        return False
    show.speak(show.lines.mood_line(mood), mood=mood)
    return hold(PERSON_FRAME, COOLDOWN_DURATION, present=True, phase="result", mood=mood)


moods = [mood.strip() for mood in args.moods.split(",") if mood.strip()]
for mood in moods if args.once else itertools.cycle(moods):
    if not greet(mood):
        break

show.close()
