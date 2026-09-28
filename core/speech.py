"""Spoken answers, plus a short chime when the assistant starts listening.

The listen cue is a system sound. Speaking the word "listening" was picked up
by the microphone and sent back to the model as if the user had said it.
Answers use a neural macOS voice when one is installed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_CHIME = "/System/Library/Sounds/Tink.aiff"
_PREFERRED_VOICES = ("Eddy (English (US))", "Flo (English (US))", "Samantha")


def _installed_voice() -> Optional[str]:
    if sys.platform != "darwin":
        return None
    try:
        listing = subprocess.check_output(["say", "-v", "?"], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    for name in _PREFERRED_VOICES:
        if name in listing:
            return name
    return None


class Speaker:
    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self._voice = _installed_voice()
        # Set while speech is audible, so the microphone can ignore it.
        self.mute_mic = threading.Event()

    def chime(self) -> None:
        if sys.platform != "darwin":
            return
        subprocess.Popen(
            ["afplay", _CHIME],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def say(self, text: str) -> None:
        phrase = " ".join(text.split())
        if not phrase:
            return
        if sys.platform != "darwin":
            logger.info("Would say: %s", phrase)
            return
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
        command = ["say", "-r", "168"]
        if self._voice:
            command.extend(["-v", self._voice])
        command.append(phrase)
        self.mute_mic.set()
        self._proc = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        threading.Thread(target=self._wait, args=(self._proc,), name="speech", daemon=True).start()

    def _wait(self, proc: subprocess.Popen) -> None:
        proc.wait()
        if self._proc is proc:
            # Leave a little tail so the last syllable is not transcribed.
            threading.Event().wait(0.35)
            if self._proc is proc:
                self.mute_mic.clear()
