"""Spoken lines of the Christmas lynx. Data only, plus a picker that avoids direct repeats.

Rules for new lines: English, no names, Christmas themed, short (the speech gate
holds the camera loop while a line plays). Write LYNQTECH as one word in capitals.
lynx_voice.PRONUNCIATION controls how `say` speaks it.
"""
import random

# Every welcome starts with the same words. The mood line follows after the analysis.
WELCOME_LINES = [
    "Welcome to the LYNQTECH Christmas party! Look at me for a moment. I will read your mood.",
    "Welcome to the LYNQTECH Christmas party! Stay right there and let me look at your face.",
    "Welcome to the LYNQTECH Christmas party! Hold still for a second. This lynx reads your mood.",
]

# Keys are the DeepFace emotion labels.
MOOD_LINES = {
    "angry": [
        "You look a bit grumpy. Careful, Santa is watching. Have a cookie.",
        "Angry at Christmas? Leave it at the door. Inside, there is only cake and good cheer.",
        "Even the Grinch came around in the end. Take a deep breath and enjoy the party.",
    ],
    "disgust": [
        "You look disgusted. Did someone bring fruitcake again?",
        "That face says somebody burned the cookies. Do not worry, there is plenty of good food inside.",
        "Not impressed? Wait until you try the Christmas treats.",
    ],
    "fear": [
        "Do not be afraid. I am a friendly lynx, and I only bite gingerbread.",
        "You look scared. Relax, the scariest thing tonight is an ugly Christmas sweater.",
        "No fear! The reindeer are parked outside and the party is safe.",
    ],
    "happy": [
        "You look happy! That is the Christmas spirit. Go and spread it.",
        "What a smile! You are on the nice list this year.",
        "You look merry and bright. The party just got better.",
    ],
    "sad": [
        "You look a little sad. A cookie and some good company will fix that.",
        "Do not be sad. It is Christmas, and something warm to drink is waiting for you.",
        "A sad face at Christmas? Come in. The lights and the music will cheer you up.",
    ],
    "surprise": [
        "You look surprised! Yes, I am a talking lynx with a Santa hat. Merry Christmas!",
        "Surprise! Christmas came early this year. Enjoy the party.",
        "Never seen a Christmas lynx before? Now you have. Come on in.",
    ],
    "neutral": [
        "A calm face. Very professional. Now relax, it is Christmas!",
        "You look neutral. Let us change that. Merry Christmas, and enjoy the party!",
        "Nice poker face! You can drop it now. Have a great evening.",
    ],
}

# For an emotion label that is not in the table.
UNKNOWN_MOOD_LINES = [
    "I cannot read your mood. Are you one of Santa's elves in disguise? Merry Christmas!",
    "Your mood is a mystery to me. Merry Christmas anyway, and enjoy the party!",
]


def all_lines():
    """Every line the lynx can say, for pre-rendering the speech."""
    lines = list(WELCOME_LINES) + list(UNKNOWN_MOOD_LINES)
    for variants in MOOD_LINES.values():
        lines.extend(variants)
    return lines


class LinePicker:
    """Picks a random variant and never the same variant twice in a row."""

    def __init__(self, rng=None):
        self._rng = rng or random.Random()
        self._last = {}

    def _pick(self, key, variants):
        choices = [v for v in variants if v != self._last.get(key)] or list(variants)
        line = self._rng.choice(choices)
        self._last[key] = line
        return line

    def welcome(self):
        return self._pick("welcome", WELCOME_LINES)

    def mood_line(self, emotion):
        key = str(emotion).lower()
        return self._pick(key, MOOD_LINES.get(key, UNKNOWN_MOOD_LINES))
