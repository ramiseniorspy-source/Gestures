from __future__ import annotations

from typing import Iterable, List, Tuple

import numpy as np
import pytest

from core.actions import Action, ActionType
from core.config import GestureConfig
from core.gesture_detector import GestureDetector, GestureState, Pose, classify, measure
from core.vision_engine import HandObservation

DT = 1.0 / 30.0
FINGER_X = (-0.04, -0.013, 0.013, 0.04)  # index, middle, ring, pinky
EXTENDED = ((0.0, -0.09), (0.0, -0.13), (0.0, -0.16), (0.0, -0.185))  # MCP, PIP, DIP, TIP
CURLED = ((0.0, -0.09), (0.0, -0.12), (0.0, -0.10), (0.0, -0.08))
THUMB_OUT = ((-0.03, -0.02), (-0.05, -0.045), (-0.07, -0.06), (-0.10, -0.06))
THUMB_IN = ((-0.02, -0.02), (-0.03, -0.045), (-0.03, -0.06), (-0.03, -0.07))


def make_hand(
    pose: str, cx: float = 0.5, cy: float = 0.5, handedness: str = "Right"
) -> HandObservation:
    """Upright synthetic hand; wrist at (cx, cy + 0.15)."""
    wrist = np.array((cx, cy + 0.15))
    fingers = {
        "palm": (True, True, True, True),
        "point": (True, False, False, False),
        "fist": (False, False, False, False),
        "pinch": (False, False, False, False),
        "scroll": (True, True, False, False),
        "three": (True, True, True, False),
    }[pose]
    pts = np.zeros((21, 3))
    pts[0, :2] = wrist
    thumb = THUMB_OUT if pose == "palm" else THUMB_IN
    for i, (dx, dy) in enumerate(thumb, start=1):
        pts[i, :2] = wrist + (dx, dy)
    for f, (x, extended) in enumerate(zip(FINGER_X, fingers)):
        for j, (_, dy) in enumerate(EXTENDED if extended else CURLED):
            pts[5 + 4 * f + j, :2] = wrist + (x, dy)
    if pose == "pinch":
        pts[6, :2] = wrist + (-0.05, -0.13)
        pts[7, :2] = wrist + (-0.06, -0.145)
        pts[8, :2] = wrist + (-0.07, -0.14)
        pts[4, :2] = wrist + (-0.068, -0.138)
    return HandObservation(
        landmarks=pts, raw_landmarks=pts, handedness=handedness, score=1.0, aspect=1.0
    )


class Runner:
    def __init__(self, config: GestureConfig = GestureConfig()) -> None:
        self.detector = GestureDetector(config)
        self.t = 100.0
        self.actions: List[Action] = []

    def feed(
        self, frames: Iterable[Tuple[str, float, float]], handedness: str = "Left"
    ) -> List[Action]:
        out: List[Action] = []
        for pose, cx, cy in frames:
            hand = None if pose == "none" else make_hand(pose, cx, cy, handedness=handedness)
            out.extend(self.detector.update(hand, self.t))
            self.t += DT
        self.actions.extend(out)
        return out

    def hold(
        self, pose: str, n: int, cx: float = 0.5, cy: float = 0.5, handedness: str = "Left"
    ) -> List[Action]:
        return self.feed([(pose, cx, cy)] * n, handedness=handedness)

    def sweep(
        self,
        pose: str,
        n: int,
        start: Tuple[float, float],
        end: Tuple[float, float],
        handedness: str = "Left",
    ) -> List[Action]:
        xs = np.linspace(start[0], end[0], n)
        ys = np.linspace(start[1], end[1], n)
        return self.feed([(pose, float(x), float(y)) for x, y in zip(xs, ys)], handedness=handedness)


def types(actions: Iterable[Action]) -> List[ActionType]:
    return [a.type for a in actions if a.type is not ActionType.MOVE_POINTER]


@pytest.mark.parametrize("pose,expected", [
    ("palm", Pose.PALM), ("point", Pose.POINTING), ("pinch", Pose.PINCH), ("fist", Pose.FIST),
    ("scroll", Pose.SCROLL), ("three", Pose.THREE),
])
def test_classify_synthetic_poses(pose: str, expected: Pose) -> None:
    config = GestureConfig()
    assert classify(measure(make_hand(pose), config), config.pinch_enter, config) is expected


def test_pointer_follows_index_tip() -> None:
    r = Runner()
    actions = r.sweep("point", 10, (0.4, 0.5), (0.6, 0.5), handedness="Right")
    moves = [a for a in actions if a.type is ActionType.MOVE_POINTER]
    assert r.detector.pointer_active
    assert len(moves) >= 5
    assert moves[-1].x > moves[0].x
    assert not types(actions)


def test_right_hand_pinch_does_not_click() -> None:
    r = Runner()
    r.hold("point", 6, handedness="Right")
    r.hold("pinch", 6, handedness="Right")
    r.hold("fist", 6, handedness="Right")
    r.hold("palm", 8, handedness="Right")
    assert not types(r.actions)


def test_quick_pinch_is_a_click() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("pinch", 5)
    r.hold("point", 6)
    assert types(r.actions) == [ActionType.CLICK]


def test_pinch_and_move_is_a_drag() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("pinch", 4)
    drag = r.sweep("pinch", 10, (0.5, 0.5), (0.65, 0.5))
    r.hold("point", 6)
    assert types(r.actions) == [ActionType.MOUSE_DOWN, ActionType.MOUSE_UP]
    moves = [a for a in drag if a.type is ActionType.MOVE_POINTER]
    assert moves and moves[-1].x > moves[0].x


def test_long_still_pinch_presses_and_holds() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("pinch", 15)
    assert types(r.actions) == [ActionType.MOUSE_DOWN]
    assert r.detector.state is GestureState.DRAGGING


def test_short_fist_is_a_right_click() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("fist", 5)
    r.hold("point", 6)
    assert types(r.actions) == [ActionType.RIGHT_CLICK]


def test_held_fist_summons_once_and_does_not_click() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("fist", 40)
    r.hold("point", 4)
    assert types(r.actions) == [ActionType.SUMMON]


def test_two_fingers_scroll_up() -> None:
    r = Runner()
    r.hold("scroll", 5, 0.5, 0.55)
    actions = r.sweep("scroll", 12, (0.5, 0.55), (0.5, 0.30))
    scrolls = [a for a in actions if a.type is ActionType.SCROLL]
    assert scrolls
    assert sum(a.amount for a in scrolls) > 0
    assert not any(a.type is ActionType.MOVE_POINTER for a in actions)


def test_three_finger_tap_is_a_double_click() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("three", 5)
    r.hold("point", 6)
    assert types(r.actions) == [ActionType.DOUBLE_CLICK]


def test_long_three_finger_hold_does_not_double_click() -> None:
    r = Runner()
    r.hold("three", 20)
    assert ActionType.DOUBLE_CLICK not in types(r.actions)


@pytest.mark.parametrize("end,expected", [
    ((0.2, 0.4), ActionType.SNAP_LEFT),
    ((0.8, 0.4), ActionType.SNAP_RIGHT),
    ((0.5, 0.75), ActionType.MINIMIZE),
    ((0.5, 0.12), ActionType.MISSION_CONTROL),
])
def test_palm_swipes(end: Tuple[float, float], expected: ActionType) -> None:
    r = Runner()
    r.hold("palm", 8, 0.5, 0.4)
    r.sweep("palm", 7, (0.5, 0.4), end)
    r.hold("palm", 10, *end)
    assert types(r.actions) == [expected]


def test_slow_palm_drift_does_not_swipe() -> None:
    r = Runner()
    r.hold("palm", 8, 0.6, 0.4)
    r.sweep("palm", 60, (0.6, 0.4), (0.3, 0.4))
    assert not types(r.actions)


def test_hand_loss_releases_drag_only_after_grace() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("pinch", 15)
    assert types(r.feed([("none", 0, 0)] * 3)) == []
    assert types(r.feed([("none", 0, 0)] * 10)) == [ActionType.MOUSE_UP]
    assert r.detector.state is GestureState.IDLE


def test_reset_during_drag_releases_mouse() -> None:
    r = Runner()
    r.hold("point", 5)
    r.hold("pinch", 15)
    assert types(r.detector.reset()) == [ActionType.MOUSE_UP]
