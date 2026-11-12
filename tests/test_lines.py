import random

import lynx_lines

DEEPFACE_EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def test_every_deepface_emotion_has_variants():
    assert sorted(lynx_lines.MOOD_LINES) == sorted(DEEPFACE_EMOTIONS)
    for emotion in DEEPFACE_EMOTIONS:
        variants = lynx_lines.MOOD_LINES[emotion]
        assert 2 <= len(variants) <= 3
        assert len(set(variants)) == len(variants)


def test_every_welcome_starts_with_the_party_welcome():
    for line in lynx_lines.WELCOME_LINES:
        assert line.startswith("Welcome to the LYNQTECH Christmas party")


def test_company_name_is_one_word_in_capitals():
    for line in lynx_lines.all_lines():
        assert "lynqtech" not in line.replace("LYNQTECH", "").lower()


def test_all_lines_covers_everything_the_picker_can_say():
    lines = lynx_lines.all_lines()
    picker = lynx_lines.LinePicker(random.Random(1))
    for _ in range(50):
        assert picker.welcome() in lines
        for emotion in DEEPFACE_EMOTIONS + ["unknown"]:
            assert picker.mood_line(emotion) in lines


def test_mood_line_matches_the_emotion():
    picker = lynx_lines.LinePicker(random.Random(2))
    for emotion in DEEPFACE_EMOTIONS:
        assert picker.mood_line(emotion) in lynx_lines.MOOD_LINES[emotion]
    assert picker.mood_line("HAPPY") in lynx_lines.MOOD_LINES["happy"]


def test_unknown_emotion_gets_a_fallback_line():
    picker = lynx_lines.LinePicker(random.Random(3))
    assert picker.mood_line("confused") in lynx_lines.UNKNOWN_MOOD_LINES


def test_picker_never_repeats_a_line_directly():
    picker = lynx_lines.LinePicker(random.Random(4))
    welcomes = [picker.welcome() for _ in range(40)]
    assert all(a != b for a, b in zip(welcomes, welcomes[1:]))
    happy = [picker.mood_line("happy") for _ in range(40)]
    assert all(a != b for a, b in zip(happy, happy[1:]))
    assert set(happy) == set(lynx_lines.MOOD_LINES["happy"])
