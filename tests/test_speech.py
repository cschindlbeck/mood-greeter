import ast
import importlib.util
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def speech_module():
    """Load speech_test.py fresh, so each test gets its own queue and worker thread."""
    spec = importlib.util.spec_from_file_location("speech_under_test", ROOT / "speech_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    module.speech_queue.put(None)
    module.speech_thread.join(timeout=2)


def test_speak_calls_say_with_rate_and_text(speech_module):
    with mock.patch.object(speech_module.subprocess, "run") as run:
        speech_module.speak("hello")
        speech_module.speech_queue.join()
    run.assert_called_once_with(["say", "-r", "195", "hello"])


def test_speak_plays_in_order(speech_module):
    with mock.patch.object(speech_module.subprocess, "run") as run:
        for text in ["one", "two", "three"]:
            speech_module.speak(text)
        speech_module.speech_queue.join()
    spoken = [call.args[0][-1] for call in run.call_args_list]
    assert spoken == ["one", "two", "three"]


def test_speak_does_not_block_on_slow_speech(speech_module):
    release = speech_module.threading.Event()
    with mock.patch.object(speech_module.subprocess, "run", side_effect=lambda *_: release.wait(2)):
        speech_module.speak("slow")
        speech_module.speak("queued")  # returns at once, although "slow" is still playing
        assert speech_module.speech_queue.unfinished_tasks == 2
        release.set()
        speech_module.speech_queue.join()


def test_none_stops_worker(speech_module):
    speech_module.speech_queue.put(None)
    speech_module.speech_thread.join(timeout=2)
    assert not speech_module.speech_thread.is_alive()


def test_is_speaking_until_last_speech_ends(speech_module):
    started = speech_module.threading.Event()
    release = speech_module.threading.Event()

    def slow_say(*_):
        started.set()
        release.wait(2)

    assert not speech_module.is_speaking()
    with mock.patch.object(speech_module.subprocess, "run", side_effect=slow_say):
        speech_module.speak("slow")
        assert started.wait(2)
        assert speech_module.speech_queue.empty()  # the worker took the item ...
        assert speech_module.is_speaking()  # ... but it still plays
        release.set()
        speech_module.speech_queue.join()
    assert not speech_module.is_speaking()


def _function_ast(path, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.dump(node)


@pytest.mark.parametrize("name", ["speech_worker", "speak", "is_speaking"])
def test_copies_stay_in_sync(name):
    """speech_worker/speak are copy-pasted in two scripts (see CLAUDE.md). Comments are ignored."""
    assert _function_ast(ROOT / "speech_test.py", name) == _function_ast(ROOT / "live_sentiment.py", name)
