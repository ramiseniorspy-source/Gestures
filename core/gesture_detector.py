"""Hand landmarks -> debounced pose -> gesture state machine -> Actions.

Two hands:
    pointer hand (GestureConfig.pointer_hand, default the right hand)
        moves the cursor from the index tip. Its pose is ignored, so a moving
        hand cannot also click, scroll, or swipe.
    the other hand
        pinch clicks or drags, a short fist right-clicks, a held fist listens,
        two fingers scroll, three fingers double-click, an open palm swipes.

Poses on the clicking hand (per frame, rotation/scale invariant):
    PINCH     thumb tip touching index tip (hysteresis on enter/exit)
    PALM      index..pinky all extended
    THREE     index, middle, and ring extended (pinky curled)
    SCROLL    index and middle extended (ring and pinky curled)
    POINTING  only the index finger extended
    FIST      every finger curled
    NONE      anything else

States:
    IDLE           no output
    POINTER        index tip drives the cursor (active region scaled to screen)
    PINCH_PENDING  pinch just began; quick release = click, move/hold = drag
    DRAGGING       mouse held; cursor follows the hand relative to the pinch origin
    PALM           swipe left/right = snap, down = minimize, up = Mission Control
    FIST_PENDING   fist just closed; quick release = right-click
    FIST_HELD      fist held still past the click window; a longer hold summons the AI
    SCROLLING      two-finger vertical motion scrolls
    THREE_PENDING  three-finger pose just began; quick release = double-click
    THREE_HELD     three fingers held past the click window; no extra action

All coordinates assume a mirrored (selfie) frame: image-left is the user's left.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Deque, List, Optional, Sequence, Tuple, Union

import numpy as np

from core.actions import Action, ActionType, assert_never
from core.config import GestureConfig
from core.vision_engine import HandObservation

WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
FINGER_JOINTS = ((6, 8), (10, 12), (14, 16), (18, 20))  # (PIP, TIP): index..pinky
PALM_POINTS = (0, 5, 9, 13, 17)

Point = Tuple[float, float]


class Pose(Enum):
    NONE = auto()
    POINTING = auto()
    PINCH = auto()
    PALM = auto()
    FIST = auto()
    SCROLL = auto()
    THREE = auto()


class GestureState(Enum):
    IDLE = auto()
    POINTER = auto()
    PINCH_PENDING = auto()
    DRAGGING = auto()
    PALM = auto()
    FIST_PENDING = auto()
    FIST_HELD = auto()
    SCROLLING = auto()
    THREE_PENDING = auto()
    THREE_HELD = auto()


@dataclass(frozen=True)
class HandFeatures:
    image_xy: np.ndarray  # (21, 2) normalized image coordinates
    iso_xy: np.ndarray  # (21, 2) x scaled by aspect: isotropic, frame-height units
    aspect: float
    palm_size: float
    pinch_ratio: float
    index_reach: float
    extended: Tuple[bool, bool, bool, bool]  # index, middle, ring, pinky


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def measure(hand: HandObservation, config: GestureConfig) -> HandFeatures:
    image_xy = hand.landmarks[:, :2]
    iso = image_xy * np.array((hand.aspect, 1.0))
    wrist = iso[WRIST]
    palm = max(_dist(wrist, iso[MIDDLE_MCP]), 1e-6)
    extended = tuple(
        _dist(wrist, iso[tip]) > _dist(wrist, iso[pip]) * config.finger_extended_ratio
        for pip, tip in FINGER_JOINTS
    )
    return HandFeatures(
        image_xy=image_xy,
        iso_xy=iso,
        aspect=hand.aspect,
        palm_size=palm,
        pinch_ratio=_dist(iso[THUMB_TIP], iso[INDEX_TIP]) / palm,
        index_reach=_dist(wrist, iso[INDEX_TIP]) / palm,
        extended=extended,  # type: ignore[arg-type]
    )


def classify(features: HandFeatures, pinch_threshold: float, config: GestureConfig) -> Pose:
    if (
        features.pinch_ratio < pinch_threshold
        and features.index_reach > config.pinch_min_index_reach
    ):
        return Pose.PINCH
    index, middle, ring, pinky = features.extended
    if index and middle and ring and pinky:
        return Pose.PALM
    if index and middle and ring and not pinky:
        return Pose.THREE
    if index and middle and not ring and not pinky:
        return Pose.SCROLL
    if index and not (middle or ring or pinky):
        return Pose.POINTING
    if not (index or middle or ring or pinky) and features.index_reach < config.fist_max_index_reach:
        return Pose.FIST
    return Pose.NONE


class GestureDetector:
    """Not thread-safe: call `update` / `reset` from a single thread (the vision thread)."""

    def __init__(self, config: GestureConfig) -> None:
        self._config = config
        self._state = GestureState.IDLE
        self._stable_pose = Pose.NONE
        self._candidate_pose = Pose.NONE
        self._candidate_frames = 0
        self._last_seen: Optional[float] = None
        self._cursor: Optional[Point] = None
        self.last_features: Optional[HandFeatures] = None
        self.pointer_active = False

        self._pinch_started = 0.0
        self._pinch_anchor = np.zeros(2)
        self._pinch_cursor: Point = (0.5, 0.5)

        self._tap_started = 0.0
        self._tap_anchor = np.zeros(2)
        self._summoned = False
        self._scroll_y: Optional[float] = None
        self._scroll_bucket = 0.0

        self._palm_armed_at = 0.0
        self._swipe_cooldown_until = 0.0
        self._palm_history: Deque[Tuple[float, np.ndarray]] = collections.deque()

    @property
    def state(self) -> GestureState:
        return self._state

    @property
    def pose(self) -> Pose:
        return self._stable_pose

    def reset(self) -> List[Action]:
        """Abort any gesture in progress (e.g. when tracking is paused)."""
        actions = self._exit_state(self._state, abort=True)
        self._state = GestureState.IDLE
        self._stable_pose = Pose.NONE
        self._candidate_pose = Pose.NONE
        self._candidate_frames = 0
        self._last_seen = None
        self._palm_history.clear()
        self._scroll_y = None
        self._scroll_bucket = 0.0
        self._summoned = False
        self.last_features = None
        return actions

    def update(
        self,
        hand: Union[HandObservation, Sequence[HandObservation], None],
        now: float,
    ) -> List[Action]:
        pointer, click = self._assign(self._coerce(hand))
        self.pointer_active = pointer is not None
        actions: List[Action] = []
        if pointer is not None:
            actions.extend(self._move_cursor(self._map_to_screen(pointer.landmarks[INDEX_TIP, :2])))
        if click is None:
            if pointer is not None:
                self.last_features = measure(pointer, self._config)
            else:
                self.last_features = None
            lost_for = now - self._last_seen if self._last_seen is not None else math.inf
            if self._state is not GestureState.IDLE and lost_for > self._config.hand_lost_grace_s:
                actions.extend(self.reset())
                self.pointer_active = pointer is not None
                if pointer is not None:
                    self.last_features = measure(pointer, self._config)
            return actions

        self._last_seen = now
        features = measure(click, self._config)
        self.last_features = features

        pinching = self._state in (GestureState.PINCH_PENDING, GestureState.DRAGGING)
        threshold = self._config.pinch_exit if pinching else self._config.pinch_enter
        pose = self._debounce(classify(features, threshold, self._config))

        actions.extend(self._set_state(self._target_state(pose), now, features))
        actions.extend(self._step(now, features))
        return actions

    def _debounce(self, raw: Pose) -> Pose:
        if raw is self._candidate_pose:
            self._candidate_frames += 1
        else:
            self._candidate_pose = raw
            self._candidate_frames = 1
        if self._candidate_frames >= self._config.pose_confirm_frames:
            self._stable_pose = raw
        return self._stable_pose

    def _target_state(self, pose: Pose) -> GestureState:
        if pose is Pose.NONE:
            return GestureState.IDLE
        if pose is Pose.POINTING:
            # Pointing is how the other hand moves. On the click hand it is idle,
            # so an extended finger does not also drag the cursor.
            return GestureState.IDLE
        if pose is Pose.PINCH:
            if self._state in (GestureState.PINCH_PENDING, GestureState.DRAGGING):
                return self._state
            return GestureState.PINCH_PENDING
        if pose is Pose.PALM:
            return GestureState.PALM
        if pose is Pose.FIST:
            if self._state in (GestureState.FIST_PENDING, GestureState.FIST_HELD):
                return self._state
            return GestureState.FIST_PENDING
        if pose is Pose.SCROLL:
            return GestureState.SCROLLING
        if pose is Pose.THREE:
            if self._state in (GestureState.THREE_PENDING, GestureState.THREE_HELD):
                return self._state
            return GestureState.THREE_PENDING
        assert_never(pose)

    def _set_state(self, new: GestureState, now: float, f: HandFeatures) -> List[Action]:
        if new is self._state:
            return []
        actions = self._exit_state(self._state, abort=False)
        self._state = new
        actions.extend(self._enter_state(new, now, f))
        return actions

    def _exit_state(self, state: GestureState, abort: bool) -> List[Action]:
        if state is GestureState.PINCH_PENDING:
            return [] if abort else [Action(ActionType.CLICK)]
        if state is GestureState.DRAGGING:
            return [Action(ActionType.MOUSE_UP)]
        if state is GestureState.PALM:
            self._palm_history.clear()
            return []
        if state is GestureState.FIST_PENDING:
            return [] if abort else [Action(ActionType.RIGHT_CLICK)]
        if state is GestureState.THREE_PENDING:
            return [] if abort else [Action(ActionType.DOUBLE_CLICK)]
        if state in (GestureState.IDLE, GestureState.POINTER, GestureState.FIST_HELD,
                     GestureState.SCROLLING, GestureState.THREE_HELD):
            return []
        assert_never(state)

    def _enter_state(self, state: GestureState, now: float, f: HandFeatures) -> List[Action]:
        if state is GestureState.PINCH_PENDING:
            actions: List[Action] = []
            if self._cursor is None:
                actions = self._move_cursor(self._map_to_screen(f.image_xy[INDEX_TIP]))
            self._pinch_started = now
            self._pinch_anchor = f.image_xy[INDEX_MCP].copy()
            self._pinch_cursor = self._cursor or (0.5, 0.5)
            return actions
        if state is GestureState.PALM:
            self._palm_armed_at = now + self._config.palm_arm_delay_s
            self._palm_history.clear()
            return []
        if state in (GestureState.FIST_PENDING, GestureState.THREE_PENDING):
            self._tap_started = now
            self._tap_anchor = f.image_xy[WRIST].copy()
            self._summoned = False
            return []
        if state is GestureState.SCROLLING:
            self._scroll_y = float(f.image_xy[MIDDLE_MCP][1])
            self._scroll_bucket = 0.0
            return []
        if state in (GestureState.IDLE, GestureState.POINTER, GestureState.DRAGGING,
                     GestureState.FIST_HELD, GestureState.THREE_HELD):
            return []
        assert_never(state)

    def _step(self, now: float, f: HandFeatures) -> List[Action]:
        state = self._state
        if state is GestureState.IDLE:
            return []
        if state is GestureState.POINTER:
            return self._move_cursor(self._map_to_screen(f.image_xy[INDEX_TIP]))
        if state is GestureState.PINCH_PENDING:
            delta = (f.image_xy[INDEX_MCP] - self._pinch_anchor) * (f.aspect, 1.0)
            moved = float(np.hypot(*delta))
            held = now - self._pinch_started
            if moved < self._config.drag_start_distance and held < self._config.click_max_duration_s:
                return []
            self._state = GestureState.DRAGGING
            if self.pointer_active:
                return [Action(ActionType.MOUSE_DOWN)]
            return [Action(ActionType.MOUSE_DOWN)] + self._drag_step(f)
        if state is GestureState.DRAGGING:
            if self.pointer_active:
                return []
            return self._drag_step(f)
        if state is GestureState.PALM:
            return self._swipe_step(now, f)
        if state is GestureState.FIST_PENDING:
            return self._fist_step(now, f)
        if state is GestureState.FIST_HELD:
            return self._fist_held_step(now, f)
        if state is GestureState.SCROLLING:
            return self._scroll_step(f)
        if state is GestureState.THREE_PENDING:
            return self._three_step(now, f)
        if state is GestureState.THREE_HELD:
            return []
        assert_never(state)

    def _drag_step(self, f: HandFeatures) -> List[Action]:
        # Follow the index knuckle, not the fingertips: it stays put while the
        # pinch tightens or loosens, so the drag does not wobble.
        x0, y0, x1, y1 = self._config.active_region
        dx, dy = f.image_xy[INDEX_MCP] - self._pinch_anchor
        target = (
            float(np.clip(self._pinch_cursor[0] + dx / (x1 - x0), 0.0, 1.0)),
            float(np.clip(self._pinch_cursor[1] + dy / (y1 - y0), 0.0, 1.0)),
        )
        return self._move_cursor(target)

    def _swipe_step(self, now: float, f: HandFeatures) -> List[Action]:
        cfg = self._config
        if now < self._palm_armed_at or now < self._swipe_cooldown_until:
            return []
        center = f.iso_xy[list(PALM_POINTS)].mean(axis=0)
        history = self._palm_history
        history.append((now, center))
        while history and now - history[0][0] > cfg.swipe_window_s:
            history.popleft()
        if len(history) < 3:
            return []

        dx, dy = center - history[0][1]
        if abs(dx) >= cfg.swipe_min_distance and abs(dx) >= cfg.swipe_axis_dominance * abs(dy):
            action = ActionType.SNAP_LEFT if dx < 0 else ActionType.SNAP_RIGHT
        elif dy >= cfg.swipe_min_distance and dy >= cfg.swipe_axis_dominance * abs(dx):
            action = ActionType.MINIMIZE
        elif -dy >= cfg.swipe_min_distance and -dy >= cfg.swipe_axis_dominance * abs(dx):
            action = ActionType.MISSION_CONTROL
        else:
            return []
        self._swipe_cooldown_until = now + cfg.swipe_cooldown_s
        history.clear()
        return [Action(action)]

    def _coerce(
        self, hand: Union[HandObservation, Sequence[HandObservation], None]
    ) -> Tuple[HandObservation, ...]:
        if hand is None:
            return ()
        if isinstance(hand, HandObservation):
            return (hand,)
        return tuple(hand)

    def _assign(
        self, hands: Sequence[HandObservation]
    ) -> Tuple[Optional[HandObservation], Optional[HandObservation]]:
        """Split hands into (pointer, click). The pointer hand never clicks."""
        if not hands:
            return None, None
        pointer_name = self._config.pointer_hand.lower()
        pointer: Optional[HandObservation] = None
        click: Optional[HandObservation] = None
        for hand in hands:
            label = hand.handedness.lower()
            if label == pointer_name and pointer is None:
                pointer = hand
            elif label in ("left", "right") and label != pointer_name and click is None:
                click = hand
        return pointer, click

    def _tap_moved(self, f: HandFeatures) -> float:
        delta = (f.image_xy[WRIST] - self._tap_anchor) * (f.aspect, 1.0)
        return float(np.hypot(*delta))

    def _fist_step(self, now: float, f: HandFeatures) -> List[Action]:
        # Leave the click window the same way a pinch becomes a drag: by assigning
        # the state here, so releasing later does not also emit the short-tap action.
        if (
            now - self._tap_started >= self._config.click_max_duration_s
            or self._tap_moved(f) >= self._config.drag_start_distance
        ):
            self._state = GestureState.FIST_HELD
            return self._fist_held_step(now, f)
        return []

    def _fist_held_step(self, now: float, f: HandFeatures) -> List[Action]:
        if self._summoned or self._tap_moved(f) >= self._config.drag_start_distance:
            self._summoned = True
            return []
        if now - self._tap_started < self._config.summon_hold_s:
            return []
        self._summoned = True
        return [Action(ActionType.SUMMON)]

    def _three_step(self, now: float, f: HandFeatures) -> List[Action]:
        if (
            now - self._tap_started >= self._config.click_max_duration_s
            or self._tap_moved(f) >= self._config.drag_start_distance
        ):
            self._state = GestureState.THREE_HELD
        return []

    def _scroll_step(self, f: HandFeatures) -> List[Action]:
        cfg = self._config
        y = float(f.image_xy[MIDDLE_MCP][1])
        if self._scroll_y is None:
            self._scroll_y = y
            return []
        dy = y - self._scroll_y
        self._scroll_y = y
        if abs(dy) < cfg.scroll_deadband:
            return []
        # Image y grows downward, so a hand moving up is a negative dy and scrolls up.
        self._scroll_bucket += -dy * cfg.scroll_gain
        amount = int(self._scroll_bucket)
        if amount == 0:
            return []
        self._scroll_bucket -= amount
        return [Action(ActionType.SCROLL, amount=amount)]

    def _map_to_screen(self, p: np.ndarray) -> Point:
        x0, y0, x1, y1 = self._config.active_region
        return (
            float(np.clip((p[0] - x0) / (x1 - x0), 0.0, 1.0)),
            float(np.clip((p[1] - y0) / (y1 - y0), 0.0, 1.0)),
        )

    def _move_cursor(self, target: Point) -> List[Action]:
        if self._cursor is not None and math.hypot(
            target[0] - self._cursor[0], target[1] - self._cursor[1]
        ) < self._config.pointer_deadband:
            return []
        self._cursor = target
        return [Action(ActionType.MOVE_POINTER, x=target[0], y=target[1])]
