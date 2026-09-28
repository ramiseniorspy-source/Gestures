"""Background voice pipeline: microphone -> wake word -> command transcription -> intent.

Runs entirely on its own thread. Audio is captured by a PortAudio callback into a
queue, so capture never drops frames while Whisper is busy transcribing.

Wake backends (VoiceConfig.backend):
    "whisper"       Energy-VAD segments speech; Whisper transcribes each segment;
                    it triggers when the utterance begins with a wake phrase
                    ("computer, snap left" in one breath, or "computer" ... pause ... command).
    "openwakeword"  Streaming neural detector on every 80 ms frame; only the command
                    after the wake word reaches Whisper. Far lower idle CPU, but needs a
                    pretrained model (alexa, hey_jarvis, hey_mycroft, ...) or a custom
                    .onnx trained for your phrase.

Recognized commands become MACRO (an input Action), CONTROL (pause/resume
gestures, dictation on/off) or AGENT_QUERY (free-form speech for the desktop
agent, which types or acts depending on what is focused).

Dictation mode listens without a wake word: each utterance is either typed or
performed. A held fist calls `request_listen`, which arms one command.
"""

from __future__ import annotations

import collections
import logging
import math
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass, replace
from enum import Enum, auto
from pathlib import Path
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from openwakeword.model import Model as WakeWordModel
from openwakeword.utils import download_models

from core.actions import Action, ActionType
from core.agent import clean_transcript
from core.config import VoiceConfig

logger = logging.getLogger(__name__)

_PRIMARY_MOD = "cmd" if sys.platform == "darwin" else "ctrl"


class IntentKind(Enum):
    MACRO = auto()
    CONTROL = auto()
    AGENT_QUERY = auto()


class ControlCommand(Enum):
    PAUSE_GESTURES = auto()
    RESUME_GESTURES = auto()
    START_DICTATION = auto()
    STOP_DICTATION = auto()


@dataclass(frozen=True)
class AgentContext:
    query: str
    screenshot_path: Optional[Path]
    captured_at: float  # wall-clock epoch seconds


@dataclass(frozen=True)
class VoiceIntent:
    kind: IntentKind
    transcript: str
    action: Optional[Action] = None
    control: Optional[ControlCommand] = None
    context: Optional[AgentContext] = None
    dictation: bool = False


def normalize_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def split_wake_phrase(
    text: str, wake_phrases: Sequence[str], max_leading_words: int = 2
) -> Optional[str]:
    """Return the command after a wake phrase ("" if bare), or None if there is none.

    The wake phrase may be preceded by up to `max_leading_words` filler words
    ("hey computer", "ok assistant").
    """
    words = normalize_text(text).split()
    for phrase in wake_phrases:
        tokens = normalize_text(phrase).split()
        for start in range(min(max_leading_words, len(words)) + 1):
            if words[start:start + len(tokens)] == tokens:
                return " ".join(words[start + len(tokens):])
    return None


_MACRO_RULES: Tuple[Tuple[str, Action], ...] = (
    (r"\b(snap|move)( (the )?window)? left\b", Action(ActionType.SNAP_LEFT)),
    (r"\b(snap|move)( (the )?window)? right\b", Action(ActionType.SNAP_RIGHT)),
    (r"^minimi[sz]e( (the )?window)?$", Action(ActionType.MINIMIZE)),
    (r"^mission control$", Action(ActionType.MISSION_CONTROL)),
    (r"^scroll up$", Action(ActionType.SCROLL, amount=5)),
    (r"^scroll down$", Action(ActionType.SCROLL, amount=-5)),
    (r"^(left )?click$", Action(ActionType.CLICK)),
    (r"^right click$", Action(ActionType.RIGHT_CLICK)),
    (r"^double click$", Action(ActionType.DOUBLE_CLICK)),
    (r"^(press |hit )?enter$|^new line$", Action(ActionType.HOTKEY, keys="enter")),
    (r"^backspace$|^delete( that)?$", Action(ActionType.HOTKEY, keys="backspace")),
    (r"^select all$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+a")),
    (r"^cut$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+x")),
    (r"^copy( that)?$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+c")),
    (r"^paste$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+v")),
    (r"^undo$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+z")),
    (r"^save( (the )?(file|document))?$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+s")),
    (r"^close( the)? (window|tab)$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+w")),
    (r"^new tab$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+t")),
    (r"^switch (app|application|window)s?$", Action(ActionType.HOTKEY, keys=f"{_PRIMARY_MOD}+tab")),
)

_CONTROL_RULES: Tuple[Tuple[str, ControlCommand], ...] = (
    (r"\b(pause|stop|disable) (gestures?|tracking)\b", ControlCommand.PAUSE_GESTURES),
    (r"\b(resume|start|enable) (gestures?|tracking)\b", ControlCommand.RESUME_GESTURES),
    (r"\b(start|begin|enable) (dictation|typing)\b", ControlCommand.START_DICTATION),
    (r"\b(take|use) (the )?keyboard\b", ControlCommand.START_DICTATION),
    (r"\b(stop|end|disable|leave|exit) (dictation|typing)\b", ControlCommand.STOP_DICTATION),
)


class IntentRouter:
    def __init__(self) -> None:
        self._macros = [(re.compile(p), a) for p, a in _MACRO_RULES]
        self._controls = [(re.compile(p), c) for p, c in _CONTROL_RULES]

    def route(self, command: str) -> VoiceIntent:
        text = normalize_text(command)
        for pattern, control in self._controls:
            if pattern.search(text):
                return VoiceIntent(IntentKind.CONTROL, command, control=control)
        for pattern, action in self._macros:
            if pattern.search(text):
                return VoiceIntent(IntentKind.MACRO, command, action=action)
        return VoiceIntent(IntentKind.AGENT_QUERY, command)


class UtteranceSegmenter:
    """Energy VAD with an adaptive noise floor. Feed float32 frames; get whole utterances."""

    def __init__(self, config: VoiceConfig) -> None:
        self._config = config
        frame_s = config.frame_samples / config.sample_rate
        self._end_frames = math.ceil(config.silence_end_s / frame_s)
        self._min_frames = math.ceil(config.min_utterance_s / frame_s)
        self._max_frames = int(config.max_utterance_s / frame_s)
        self._preroll: Deque[np.ndarray] = collections.deque(maxlen=config.vad_preroll_frames)
        self._frames: List[np.ndarray] = []
        self._noise_floor = config.vad_min_rms / config.vad_noise_ratio
        self._loud_run = 0
        self._silent_run = 0
        self.in_speech = False

    def reset(self) -> None:
        self._preroll.clear()
        self._frames = []
        self._loud_run = 0
        self._silent_run = 0
        self.in_speech = False

    def push(self, frame: np.ndarray) -> Optional[np.ndarray]:
        cfg = self._config
        rms = float(np.sqrt(np.mean(frame * frame)))
        loud = rms > max(cfg.vad_min_rms, self._noise_floor * cfg.vad_noise_ratio)

        if not self.in_speech:
            self._preroll.append(frame)
            if loud:
                self._loud_run += 1
            else:
                self._loud_run = 0
                self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
            if self._loud_run >= cfg.vad_start_frames:
                self.in_speech = True
                self._frames = list(self._preroll)
                self._preroll.clear()
                self._silent_run = 0
            return None

        self._frames.append(frame)
        self._silent_run = 0 if loud else self._silent_run + 1
        if self._silent_run < self._end_frames and len(self._frames) < self._max_frames:
            return None

        audio = np.concatenate(self._frames)
        voiced_frames = len(self._frames) - self._silent_run
        self.reset()
        return audio if voiced_frames >= self._min_frames else None


class Transcriber:
    def __init__(self, config: VoiceConfig) -> None:
        logger.info("Loading Whisper model %r", config.whisper_model)
        self._model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=config.whisper_compute_type,
        )

    def __call__(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(
            audio,
            language="en",
            beam_size=1,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


class WakeWordSpotter:
    def __init__(self, config: VoiceConfig) -> None:
        model = config.openwakeword_model
        if not Path(model).exists():
            download_models([model])
        self._model = WakeWordModel(wakeword_models=[model], inference_framework="onnx")
        self._threshold = config.openwakeword_threshold

    def detected(self, frame: np.ndarray) -> bool:
        scores = self._model.predict(frame)
        return max(scores.values(), default=0.0) >= self._threshold

    def reset(self) -> None:
        self._model.reset()


class _Phase(Enum):
    WAITING_FOR_WAKE = auto()
    AWAITING_COMMAND = auto()


class VoiceListener:
    def __init__(
        self,
        config: VoiceConfig,
        on_intent: Callable[[VoiceIntent], None],
        on_notice: Optional[Callable[[str], None]] = None,
        mute_mic: Optional[threading.Event] = None,
    ) -> None:
        if config.backend not in ("whisper", "openwakeword"):
            raise ValueError(f"Unknown voice backend: {config.backend!r}")
        self._config = config
        self._on_intent = on_intent
        self._on_notice = on_notice
        self._mute_mic = mute_mic
        self._router = IntentRouter()
        self._stop = threading.Event()
        self._dictating = threading.Event()
        self._dictation_on = threading.Event()
        self._dictation_off = threading.Event()
        self._arm_listen = threading.Event()
        self.phase_name = "starting"
        self._audio: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)
        self._thread = threading.Thread(target=self._run, name="voice", daemon=True)
        self.error: Optional[BaseException] = None

    def set_dictation(self, enabled: bool) -> None:
        """Switch keyboard-replacement mode. Applied on the voice thread."""
        if enabled:
            self._dictation_off.clear()
            self._dictation_on.set()
            self.phase_name = "dictation"
        else:
            self._dictation_on.clear()
            self._dictation_off.set()
            self.phase_name = "command"

    def request_listen(self) -> None:
        """Arm one command with no wake word. A held fist uses this."""
        self._arm_listen.set()
        self.phase_name = "listening"

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            logger.debug("Audio status: %s", status)
        try:
            self._audio.put_nowait(indata[:, 0].copy())
        except queue.Full:
            pass

    def _run(self) -> None:
        cfg = self._config
        try:
            transcriber = Transcriber(cfg)
            spotter = WakeWordSpotter(cfg) if cfg.backend == "openwakeword" else None
            segmenter = UtteranceSegmenter(cfg)
            if self._stop.is_set():
                return
            logger.info("Opening microphone (macOS blocks here until Microphone permission "
                        "is granted to this app)")
            with sd.InputStream(
                samplerate=cfg.sample_rate,
                channels=1,
                dtype="int16",
                blocksize=cfg.frame_samples,
                callback=self._audio_callback,
            ):
                wake = cfg.openwakeword_model if spotter else " / ".join(cfg.wake_phrases)
                logger.info("Voice listener ready (%s backend); wake word: %s", cfg.backend, wake)
                self._loop(transcriber, spotter, segmenter)
        except Exception as exc:
            self.error = exc
            logger.exception("Voice listener stopped")

    def _loop(
        self,
        transcriber: Transcriber,
        spotter: Optional[WakeWordSpotter],
        segmenter: UtteranceSegmenter,
    ) -> None:
        cfg = self._config
        phase = _Phase.WAITING_FOR_WAKE
        deadline = 0.0
        zero_frames = 0
        zero_limit = int(3.0 * cfg.sample_rate / cfg.frame_samples)
        was_muted = False

        def back_to_waiting() -> _Phase:
            segmenter.reset()
            if spotter is not None:
                spotter.reset()
            return _Phase.WAITING_FOR_WAKE

        while not self._stop.is_set():
            try:
                frame = self._audio.get(timeout=0.2)
            except queue.Empty:
                continue
            now = time.monotonic()
            if self._mute_mic is not None and self._mute_mic.is_set():
                if not was_muted:
                    segmenter.reset()
                    was_muted = True
                continue
            if was_muted:
                segmenter.reset()
                was_muted = False
            if self._dictation_on.is_set():
                self._dictation_on.clear()
                self._dictating.set()
                segmenter.reset()
                phase = back_to_waiting()
                logger.info("Dictation on; speech will be typed or performed")
            if self._dictation_off.is_set():
                self._dictation_off.clear()
                self._dictating.clear()
                segmenter.reset()
                phase = back_to_waiting()
                logger.info("Dictation off; say a wake word to call the assistant")
            dictating = self._dictating.is_set()
            if self._arm_listen.is_set() and not dictating:
                self._arm_listen.clear()
                segmenter.reset()
                if spotter is not None:
                    spotter.reset()
                phase, deadline = _Phase.AWAITING_COMMAND, now + max(cfg.command_timeout_s, 8.0)
                logger.info("Listening for a command")
                self._notice("Listening")

            # macOS delivers digital silence instead of an error when the app lacks
            # Microphone permission.
            zero_frames = zero_frames + 1 if not frame.any() else 0
            if zero_frames == zero_limit:
                logger.warning("Microphone is delivering pure silence; check the input "
                               "device and Microphone permission for this app")

            if phase is _Phase.AWAITING_COMMAND and now > deadline and not segmenter.in_speech:
                logger.info("No command heard")
                phase = back_to_waiting()
                self._notice("I didn't hear anything")

            self.phase_name = (
                "dictation" if dictating
                else "listening" if phase is _Phase.AWAITING_COMMAND
                else "command"
            )

            if spotter is not None and not dictating and phase is _Phase.WAITING_FOR_WAKE:
                if spotter.detected(frame):
                    logger.info("Wake word detected; listening for command")
                    segmenter.reset()
                    phase, deadline = _Phase.AWAITING_COMMAND, now + cfg.command_timeout_s
                    self._notice("Listening")
                continue

            utterance = segmenter.push(frame.astype(np.float32) / 32768.0)
            if utterance is None:
                continue
            text = transcriber(utterance)
            if not text:
                continue
            logger.debug("Heard: %r", text)

            remainder = split_wake_phrase(text, cfg.wake_phrases)
            if dictating:
                spoken = text if remainder is None else remainder
                if spoken:
                    self._dispatch(spoken, dictation=True)
                continue
            if phase is _Phase.AWAITING_COMMAND:
                if remainder == "":
                    deadline = now + cfg.command_timeout_s
                    continue
                phase = back_to_waiting()
                self._dispatch(remainder if remainder is not None else text, dictation=False)
            elif remainder:
                self._dispatch(remainder, dictation=False)
            elif remainder is not None:
                logger.info("Wake word detected; listening for command")
                phase, deadline = _Phase.AWAITING_COMMAND, now + cfg.command_timeout_s
                self._notice("Listening")

    def _notice(self, text: str) -> None:
        if self._on_notice is None:
            return
        try:
            self._on_notice(text)
        except Exception:
            logger.exception("Voice notice failed")

    def _dispatch(self, command: str, *, dictation: bool) -> None:
        command = clean_transcript(command)
        if not command:
            logger.info("Ignored the assistant's own listening cue")
            return
        intent = self._router.route(command)
        if intent.kind is IntentKind.AGENT_QUERY:
            # Focus (which app and control) is what the agent uses to choose typing
            # versus an action. Screenshot capture stays off this thread.
            context = AgentContext(command, None, time.time())
            intent = replace(intent, context=context, dictation=dictation)
        else:
            intent = replace(intent, dictation=dictation)
        logger.info("Voice intent %s: %r", intent.kind.name, command)
        try:
            self._on_intent(intent)
        except Exception:
            logger.exception("Voice intent handler failed")
