"""Touchless control script: hand gestures, voice, and a local desktop agent.

Run it while you want it (it is not a login item):

    python main.py
    python main.py --headless
    python main.py --dry-run -v

Gestures (one hand, selfie view):
    right hand            move the pointer. Its pose is ignored.
    left hand pinch       click; hold it and move the right hand to drag
    left hand, two fingers  scroll
    left fist tap         right-click
    left fist held still  listen, then the assistant answers out loud
    left three-finger tap double-click
    left palm swipe       left/right snap, down minimize, up Mission Control

Voice:
    Click a text field and the next things you say are typed there.
    "computer, ..." / a held left fist   the assistant thinks and answers aloud
    "enter", "backspace", "select all"   still keys, including while dictating

The agent is a local model (Ollama llama3.1:8b by default). It sees the phrase
and whether a text field is focused, then types, presses keys, or both.
"--no-agent" keeps the same choices without calling the model.

Threads:
    main    orchestration, signal handling, and the optional OpenCV debug window
    vision  camera -> MediaPipe -> smoothing -> gesture state machine
    input   mouse, keys, and typed text
    voice   microphone -> wake word or dictation -> intents
    agent   model (or the offline decider) turns a phrase into actions

macOS permissions for the app running Python (Terminal, iTerm, Cursor, ...):
Camera, Accessibility (synthetic input and the focused control), and Microphone.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import queue
import signal
import sys
import threading
import time
from typing import Optional

import cv2

from core.actions import Action, ActionType, assert_never
from core.agent import DesktopAgent, dictation_text
from core.config import AgentConfig, AppConfig
from core.focus import read_focus
from core.gesture_detector import GestureDetector
from core.input_controller import InputController
from core.overlay import WINDOW_NAME, draw_hud
from core.speech import Speaker
from core.vision_engine import VisionEngine, VisionFrame

try:
    from core.voice_listener import (
        ControlCommand,
        IntentKind,
        VoiceIntent,
        VoiceListener,
    )

    VOICE_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:  # voice extras are optional (requirements-voice.txt)
    VoiceListener = None  # type: ignore[assignment,misc]
    VOICE_IMPORT_ERROR = exc

logger = logging.getLogger("main")


class Orchestrator:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._detector_suspended = False

        self._input = InputController(config.input)
        self._detector = GestureDetector(config.gesture)
        self._vision = VisionEngine(
            config.vision, on_frame=self._on_vision_frame, render_debug=config.show_debug
        )
        self._agent = DesktopAgent(config.agent)
        self._agent_queue: "queue.Queue[Optional[VoiceIntent]]" = queue.Queue(maxsize=4)
        self._agent_note = ""
        self._speaker = Speaker()
        self._manual_dictation = False
        self._focus_checked = 0.0
        self._agent_thread = threading.Thread(target=self._agent_loop, name="agent", daemon=True)
        self._voice: Optional["VoiceListener"] = None
        if config.enable_voice:
            if VoiceListener is None:
                logger.warning("Voice disabled, missing dependency: %s "
                               "(pip install -r requirements-voice.txt)", VOICE_IMPORT_ERROR)
            else:
                self._voice = VoiceListener(
                    config.voice,
                    on_intent=self._on_voice_intent,
                    on_notice=self._on_notice,
                    mute_mic=self._speaker.mute_mic,
                )

    def request_stop(self, *_: object) -> None:
        self._stop.set()

    def run(self) -> int:
        self._input.start()
        self._agent_thread.start()
        self._vision.start()
        try:
            self._vision.wait_ready()
            if self._vision.error is not None:
                logger.error("Vision failed to start: %s", self._vision.error)
                return 1
            if self._voice is not None:
                self._voice.start()
            logger.info("Running. %s", "Press q/Esc in the window to quit."
                        if self._config.show_debug else "Ctrl+C to quit.")

            if self._config.show_debug:
                self._display_loop()
            else:
                while not self._stop.is_set() and self._vision.is_alive():
                    self._sync_dictation()
                    self._stop.wait(0.25)
            return 1 if self._vision.error is not None else 0
        finally:
            self._shutdown()

    def _shutdown(self) -> None:
        self._stop.set()
        self._vision.stop()
        self._vision.join(timeout=3.0)
        if self._voice is not None:
            self._voice.stop()
            if self._voice.is_alive():
                self._voice.join(timeout=3.0)
        for action in self._detector.reset():
            if action.type is not ActionType.SUMMON:
                self._input.submit(action)
        self._input.stop()
        self._input.join(timeout=3.0)
        if self._config.show_debug:
            cv2.destroyAllWindows()
        self._agent_thread.join(timeout=1.0)
        logger.info("Stopped")

    def _display_loop(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while not self._stop.is_set() and self._vision.is_alive():
            self._sync_dictation()
            image = self._vision.latest_debug_image()
            if image is not None:
                click = "PAUSED" if self._paused.is_set() else self._detector.pose.name
                move = "right hand moving" if self._detector.pointer_active else "show your right hand to move"
                voice = self._voice.phase_name if self._voice is not None else "off"
                lines = [f"{move}  left hand {click}", f"voice {voice}"]
                if self._agent_note:
                    lines.append(self._agent_note)
                draw_hud(image, self._config.gesture, self._detector.last_features, lines)
                cv2.imshow(WINDOW_NAME, image)
            if (cv2.waitKey(10) & 0xFF) in (ord("q"), 27):
                self._stop.set()

    def _on_vision_frame(self, frame: VisionFrame) -> None:
        # Vision thread. The detector is only ever touched here (and in shutdown,
        # after the vision thread has been joined).
        if self._paused.is_set():
            if not self._detector_suspended:
                for action in self._detector.reset():
                    self._input.submit(action)
                self._detector_suspended = True
            return
        self._detector_suspended = False
        for action in self._detector.update(frame.hands, frame.timestamp):
            if action.type is ActionType.SUMMON:
                self._summon_agent()
            else:
                self._input.submit(action)

    def _on_voice_intent(self, intent: "VoiceIntent") -> None:
        # Voice thread: every branch must be non-blocking.
        kind = intent.kind
        if kind is IntentKind.MACRO:
            if intent.action is not None:
                self._input.submit(intent.action)
        elif kind is IntentKind.CONTROL:
            control = intent.control
            if control is ControlCommand.PAUSE_GESTURES:
                self._paused.set()
                logger.info("Gesture control paused")
            elif control is ControlCommand.RESUME_GESTURES:
                self._paused.clear()
                logger.info("Gesture control resumed")
            elif control is ControlCommand.START_DICTATION:
                self._manual_dictation = True
                if self._voice is not None:
                    self._voice.set_dictation(True)
            elif control is ControlCommand.STOP_DICTATION:
                self._manual_dictation = False
                if self._voice is not None:
                    self._voice.set_dictation(False)
            elif control is None:
                pass
            else:
                assert_never(control)
        elif kind is IntentKind.AGENT_QUERY:
            if intent.dictation:
                typed = dictation_text(intent.transcript)
                if typed:
                    self._input.submit(Action(ActionType.TYPE_TEXT, text=typed))
                    self._agent_note = "typed"
                    logger.info("Dictated %r", typed)
                return
            self._enqueue_agent(intent)
        else:
            assert_never(kind)

    def _summon_agent(self) -> None:
        if self._voice is None:
            logger.info("Fist held, but voice is off")
            return
        self._voice.request_listen()

    def _on_notice(self, text: str) -> None:
        self._agent_note = text
        if text == "Listening":
            self._speaker.chime()
            return
        # A spoken "I didn't hear anything" would be picked up by the microphone.

    def _sync_dictation(self) -> None:
        """A focused text field is the keyboard: speech there is typed."""
        if self._voice is None:
            return
        now = time.monotonic()
        if now - self._focus_checked < 0.35:
            return
        self._focus_checked = now
        focused = read_focus()
        dictating = self._voice.phase_name == "dictation"
        if focused.editable and not dictating:
            self._voice.set_dictation(True)
            where = focused.app_name or "this field"
            self._agent_note = f"typing in {where}"
            logger.info("Text field focused (%s %s); dictation on", focused.app_name, focused.role)
        elif not focused.editable and dictating and not self._manual_dictation:
            self._voice.set_dictation(False)
            logger.info("Left the text field; dictation off")

    def _enqueue_agent(self, intent: "VoiceIntent") -> None:
        try:
            self._agent_queue.put_nowait(intent)
        except queue.Full:
            logger.warning("Agent is busy; dropped %r", intent.transcript)

    def _agent_loop(self) -> None:
        while not self._stop.is_set():
            try:
                intent = self._agent_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if intent is None:
                return
            self._run_agent(intent)

    def _run_agent(self, intent: "VoiceIntent") -> None:
        focus = read_focus()
        phrase = intent.context.query if intent.context is not None else intent.transcript
        where = focus.app_name or "no app"
        self._agent_note = "thinking"
        logger.info(
            "Agent heard %r (%s, %s, editable=%s, dictation=%s)",
            phrase, where, focus.role or "no control", focus.editable, intent.dictation,
        )
        actions, spoken = self._agent.decide(phrase, focus, dictation=intent.dictation)
        self._agent_note = spoken[:80]
        if spoken:
            self._speaker.say(spoken)
        if not actions:
            logger.info("Agent spoke and took no further action (%s)", spoken)
            return
        logger.info("Agent actions: %s (%s)", ", ".join(_describe(a) for a in actions), spoken)
        for action in actions:
            if (
                intent.dictation
                and action.type is ActionType.TYPE_TEXT
                and action.text
                and not action.text[-1].isspace()
            ):
                action = dataclasses.replace(action, text=action.text + " ")
            self._input.submit(action)


def _describe(action: Action) -> str:
    if action.type is ActionType.TYPE_TEXT:
        return f"type {action.text!r}"
    if action.type is ActionType.HOTKEY:
        return action.keys
    if action.type is ActionType.SCROLL:
        return f"scroll {action.amount}"
    return action.type.name.lower()


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--headless", action="store_true", help="no debug window")
    parser.add_argument("--no-voice", action="store_true", help="disable the voice listener")
    parser.add_argument("--dry-run", action="store_true", help="log actions instead of performing them")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--voice-backend", choices=("whisper", "openwakeword"), default="whisper")
    parser.add_argument("--wake-model", default=None,
                        help="openWakeWord model name or .onnx path (openwakeword backend)")
    parser.add_argument("--whisper-model", default=None, help="e.g. tiny.en, base.en, small.en")
    parser.add_argument("--no-agent", action="store_true",
                        help="decide type-versus-action locally, without calling the model")
    parser.add_argument("--agent-model", default=None, help="Ollama (or other) chat model name")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> AppConfig:
    base = AppConfig()
    voice = dataclasses.replace(base.voice, backend=args.voice_backend)
    if args.wake_model:
        voice = dataclasses.replace(voice, openwakeword_model=args.wake_model)
    if args.whisper_model:
        voice = dataclasses.replace(voice, whisper_model=args.whisper_model)
    agent = AgentConfig(
        base_url=os.environ.get("GESTURE_AI_BASE_URL", base.agent.base_url),
        model=args.agent_model or os.environ.get("GESTURE_AI_MODEL", base.agent.model),
        api_key=os.environ.get("GESTURE_AI_API_KEY", base.agent.api_key),
        timeout_s=base.agent.timeout_s,
        enabled=not args.no_agent,
    )
    return dataclasses.replace(
        base,
        vision=dataclasses.replace(base.vision, camera_index=args.camera),
        input=dataclasses.replace(base.input, dry_run=args.dry_run),
        voice=voice,
        agent=agent,
        show_debug=not args.headless,
        enable_voice=not args.no_voice,
    )


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(threadName)-7s %(levelname)-7s %(name)s: %(message)s",
    )
    orchestrator = Orchestrator(build_config(args))
    signal.signal(signal.SIGINT, orchestrator.request_stop)
    signal.signal(signal.SIGTERM, orchestrator.request_stop)
    return orchestrator.run()


if __name__ == "__main__":
    sys.exit(main())
