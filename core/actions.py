"""Shared action contract between producers (gestures, voice) and the input controller."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import NoReturn


class ActionType(Enum):
    MOVE_POINTER = auto()  # x, y: normalized screen coordinates in [0, 1]
    MOUSE_DOWN = auto()
    MOUSE_UP = auto()
    CLICK = auto()
    RIGHT_CLICK = auto()
    DOUBLE_CLICK = auto()
    SCROLL = auto()  # amount: positive scrolls up
    HOTKEY = auto()  # keys: chord spec such as "cmd+shift+4", or a single key
    TYPE_TEXT = auto()  # text: characters to insert at the cursor
    SNAP_LEFT = auto()
    SNAP_RIGHT = auto()
    MINIMIZE = auto()
    MISSION_CONTROL = auto()
    SUMMON = auto()  # hold-to-listen; the voice thread handles this, not the OS


@dataclass(frozen=True)
class Action:
    type: ActionType
    x: float = 0.0
    y: float = 0.0
    amount: int = 0
    keys: str = ""
    text: str = ""


def assert_never(value: NoReturn) -> NoReturn:
    raise AssertionError(f"Unhandled value: {value!r}")
