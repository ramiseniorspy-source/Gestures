"""OS input synthesis on a dedicated worker thread.

Producers call `submit()`, which never blocks: actions go onto a bounded queue
drained by the worker. Consecutive pointer moves are coalesced so a slow OS call
can never build up cursor lag, while discrete actions (down/up/click/hotkeys)
keep their exact order relative to moves.

Mouse events go through pynput because it emits real drag events while a
button is held (pyautogui's moveTo does not on macOS). pyautogui supplies the
screen size.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
from typing import List, Optional, Protocol, Union

import pyautogui
from pynput import keyboard, mouse

from core.actions import Action, ActionType, assert_never
from core.config import InputConfig

if sys.platform == "darwin":
    import Quartz

logger = logging.getLogger(__name__)

_KEY_ALIASES = {
    "control": "ctrl",
    "command": "cmd",
    "win": "cmd",
    "super": "cmd",
    "meta": "cmd",
    "option": "alt",
    "globe": "fn",
    "return": "enter",
    "escape": "esc",
}


def parse_chord(spec: str) -> List[str]:
    names = [_KEY_ALIASES.get(part.strip().lower(), part.strip().lower()) for part in spec.split("+")]
    if not names or any(not n for n in names):
        raise ValueError(f"Invalid key chord: {spec!r}")
    return names


def _resolve_pynput_key(name: str) -> Union[keyboard.Key, str]:
    if len(name) == 1:
        return name
    key = getattr(keyboard.Key, name, None)
    if key is None:
        raise ValueError(f"Unknown key name: {name!r}")
    return key


class _Backend(Protocol):
    def move(self, x: float, y: float) -> None: ...
    def mouse_down(self) -> None: ...
    def mouse_up(self) -> None: ...
    def click(self) -> None: ...
    def right_click(self) -> None: ...
    def double_click(self) -> None: ...
    def scroll(self, amount: int) -> None: ...
    def hotkey(self, spec: str) -> None: ...
    def type_text(self, text: str) -> None: ...
    def release_all(self) -> None: ...


class _PynputBackend:
    def __init__(self) -> None:
        self._mouse = mouse.Controller()
        self._keyboard = keyboard.Controller()
        width, height = pyautogui.size()  # primary display, in OS points
        self._max_x = width - 1
        self._max_y = height - 1
        self._button_held = False

    def move(self, x: float, y: float) -> None:
        self._mouse.position = (round(x * self._max_x), round(y * self._max_y))

    def mouse_down(self) -> None:
        if not self._button_held:
            self._mouse.press(mouse.Button.left)
            self._button_held = True

    def mouse_up(self) -> None:
        if self._button_held:
            self._mouse.release(mouse.Button.left)
            self._button_held = False

    def click(self) -> None:
        self._mouse.click(mouse.Button.left)

    def right_click(self) -> None:
        self._mouse.click(mouse.Button.right)

    def double_click(self) -> None:
        self._mouse.click(mouse.Button.left, 2)

    def scroll(self, amount: int) -> None:
        self._mouse.scroll(0, amount)

    def hotkey(self, spec: str) -> None:
        names = parse_chord(spec)
        if sys.platform == "darwin" and "fn" in names:
            _post_mac_chord(names)
            return
        keys = [_resolve_pynput_key(n) for n in names]
        pressed = []
        try:
            for key in keys:
                self._keyboard.press(key)
                pressed.append(key)
        finally:
            for key in reversed(pressed):
                self._keyboard.release(key)

    def type_text(self, text: str) -> None:
        if text:
            self._keyboard.type(text)

    def release_all(self) -> None:
        self.mouse_up()


class _DryRunBackend:
    def move(self, x: float, y: float) -> None:
        logger.debug("[dry-run] move %.3f, %.3f", x, y)

    def mouse_down(self) -> None:
        logger.info("[dry-run] mouse down")

    def mouse_up(self) -> None:
        logger.info("[dry-run] mouse up")

    def click(self) -> None:
        logger.info("[dry-run] click")

    def right_click(self) -> None:
        logger.info("[dry-run] right click")

    def double_click(self) -> None:
        logger.info("[dry-run] double click")

    def scroll(self, amount: int) -> None:
        logger.info("[dry-run] scroll %d", amount)

    def hotkey(self, spec: str) -> None:
        parse_chord(spec)
        logger.info("[dry-run] hotkey %s", spec)

    def type_text(self, text: str) -> None:
        logger.info("[dry-run] type %r", text)

    def release_all(self) -> None:
        pass


if sys.platform == "darwin":
    _MAC_KEYCODES = {"left": 0x7B, "right": 0x7C, "down": 0x7D, "up": 0x7E}
    _MAC_FLAGS = {
        "ctrl": Quartz.kCGEventFlagMaskControl,
        "shift": Quartz.kCGEventFlagMaskShift,
        "alt": Quartz.kCGEventFlagMaskAlternate,
        "cmd": Quartz.kCGEventFlagMaskCommand,
        "fn": Quartz.kCGEventFlagMaskSecondaryFn,
    }

    def _post_mac_chord(names: List[str]) -> None:
        # The fn/Globe modifier can't be synthesized as a normal key press; macOS
        # menu shortcuts (e.g. Window > Move & Resize) match on event flags, so the
        # modifiers are attached to the key event directly.
        *modifiers, key = names
        if key not in _MAC_KEYCODES:
            raise ValueError(f"fn chords support only arrow keys, got {key!r}")
        flags = 0
        for name in modifiers:
            if name not in _MAC_FLAGS:
                raise ValueError(f"Unknown modifier in fn chord: {name!r}")
            flags |= _MAC_FLAGS[name]
        source = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateHIDSystemState)
        for is_down in (True, False):
            event = Quartz.CGEventCreateKeyboardEvent(source, _MAC_KEYCODES[key], is_down)
            Quartz.CGEventSetFlags(event, flags)
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)

else:

    def _post_mac_chord(names: List[str]) -> None:
        raise RuntimeError("fn chords are only supported on macOS")


def _coalesce_moves(batch: List[Optional[Action]]) -> List[Optional[Action]]:
    out: List[Optional[Action]] = []
    for action in batch:
        is_move = action is not None and action.type is ActionType.MOVE_POINTER
        prev = out[-1] if out else None
        if is_move and prev is not None and prev.type is ActionType.MOVE_POINTER:
            out[-1] = action
        else:
            out.append(action)
    return out


class InputController:
    def __init__(self, config: InputConfig) -> None:
        self._config = config
        self._queue: "queue.Queue[Optional[Action]]" = queue.Queue(maxsize=config.queue_size)
        self._backend: _Backend = _DryRunBackend() if config.dry_run else _PynputBackend()
        self._thread = threading.Thread(target=self._run, name="input", daemon=True)
        self._dropped = 0
        for spec in (config.snap_left_keys, config.snap_right_keys, config.minimize_keys,
                     config.mission_control_keys):
            parse_chord(spec)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    def submit(self, action: Action) -> None:
        try:
            self._queue.put_nowait(action)
        except queue.Full:
            self._dropped += 1
            if self._dropped % 100 == 1:
                logger.warning("Input queue full; dropped %d actions so far", self._dropped)

    def _run(self) -> None:
        try:
            while True:
                batch: List[Optional[Action]] = [self._queue.get()]
                while True:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break
                for action in _coalesce_moves(batch):
                    if action is None:
                        return
                    try:
                        self._execute(action)
                    except Exception:
                        logger.exception("Failed to execute %s", action)
        finally:
            self._backend.release_all()

    def _execute(self, action: Action) -> None:
        kind = action.type
        backend = self._backend
        if kind is ActionType.MOVE_POINTER:
            backend.move(action.x, action.y)
        elif kind is ActionType.MOUSE_DOWN:
            backend.mouse_down()
        elif kind is ActionType.MOUSE_UP:
            backend.mouse_up()
        elif kind is ActionType.CLICK:
            backend.click()
        elif kind is ActionType.RIGHT_CLICK:
            backend.right_click()
        elif kind is ActionType.DOUBLE_CLICK:
            backend.double_click()
        elif kind is ActionType.SCROLL:
            backend.scroll(action.amount)
        elif kind is ActionType.HOTKEY:
            backend.hotkey(action.keys)
        elif kind is ActionType.TYPE_TEXT:
            backend.type_text(action.text)
        elif kind is ActionType.SNAP_LEFT:
            backend.hotkey(self._config.snap_left_keys)
        elif kind is ActionType.SNAP_RIGHT:
            backend.hotkey(self._config.snap_right_keys)
        elif kind is ActionType.MINIMIZE:
            backend.hotkey(self._config.minimize_keys)
        elif kind is ActionType.MISSION_CONTROL:
            backend.hotkey(self._config.mission_control_keys)
        elif kind is ActionType.SUMMON:
            logger.debug("Summon is handled by the voice listener")
        else:
            assert_never(kind)
