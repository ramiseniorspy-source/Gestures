"""Tunable parameters for every subsystem. All distances are documented with their units."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CAPTURES_DIR = PROJECT_ROOT / "captures"

HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass(frozen=True)
class VisionConfig:
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    max_fps: float = 60.0
    # Flip horizontally so image-left == user's left (selfie view). Gesture
    # directions assume this is enabled.
    mirror: bool = True
    model_path: Path = MODELS_DIR / "hand_landmarker.task"
    model_url: str = HAND_LANDMARKER_URL
    # On macOS the CPU delegate aborts in mediapipe >= 0.10.30 (its detection
    # calculator unconditionally requests a Metal service), so Metal is used there.
    # MediaPipe's Python GPU delegate is unsupported on Windows.
    use_gpu: bool = sys.platform == "darwin"
    min_detection_confidence: float = 0.6
    min_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    # One Euro filter on normalized landmark coordinates. Lower min_cutoff = less
    # jitter at rest; higher beta = less lag during fast motion.
    smoothing_min_cutoff: float = 1.5
    smoothing_beta: float = 5.0
    smoothing_d_cutoff: float = 1.0
    reopen_after_failed_reads: int = 30


@dataclass(frozen=True)
class GestureConfig:
    # Thumb-tip to index-tip distance divided by palm size (wrist -> middle MCP).
    # Hysteresis: enter below `pinch_enter`, release only above `pinch_exit`.
    pinch_enter: float = 0.28
    pinch_exit: float = 0.42
    # Wrist -> index-tip distance / palm size must exceed this for a pinch, so a
    # closed fist (thumb resting near a curled index tip) is not read as a click.
    pinch_min_index_reach: float = 1.1
    # A finger is extended when wrist->tip exceeds wrist->PIP by this factor.
    finger_extended_ratio: float = 1.22
    # A raw pose must persist this many frames before the state machine accepts it.
    # High enough that a finger flickering between poses does not click or swipe.
    pose_confirm_frames: int = 4
    # The hand that moves the cursor. The other hand clicks, scrolls, and swipes.
    pointer_hand: str = "Right"
    # Camera sub-rectangle (x0, y0, x1, y1 in normalized image coords) mapped onto
    # the full screen, so the hand never has to reach the frame edges.
    active_region: Tuple[float, float, float, float] = (0.15, 0.12, 0.85, 0.72)
    # Minimum cursor change (normalized screen units) before a move is emitted.
    pointer_deadband: float = 0.0015
    # Pinch shorter than this with little movement is a click; longer starts a drag.
    click_max_duration_s: float = 0.30
    # Hand movement (frame-height units) during a pinch that promotes it to a drag.
    drag_start_distance: float = 0.025
    # Palm swipe: displacement (frame-height units) within the sliding window.
    swipe_window_s: float = 0.35
    swipe_min_distance: float = 0.26
    swipe_axis_dominance: float = 2.0
    swipe_cooldown_s: float = 0.9
    # Ignore palm motion right after the pose appears (hand entering the frame).
    palm_arm_delay_s: float = 0.15
    # Keep state (e.g. a held drag) through brief tracking dropouts.
    hand_lost_grace_s: float = 0.25
    # A closed fist (every finger curled) must be this short, wrist to index tip
    # over palm size, so a half-open hand is not a right-click.
    fist_max_index_reach: float = 1.05
    # Hold a still fist this long to listen for one AI command. Shorter is a right-click.
    summon_hold_s: float = 0.70
    # Two-finger scroll: cursor-wheel clicks per full frame-height of hand travel.
    scroll_gain: float = 80.0
    scroll_deadband: float = 0.012


def _default_hotkeys() -> Tuple[str, str, str, str]:
    """(snap_left, snap_right, minimize, mission_control) for the host OS."""
    if sys.platform == "darwin":
        # macOS 15+ native tiling (Globe/fn + Control + arrow); Cmd+M minimizes.
        # Control+Up opens Mission Control.
        return "ctrl+fn+left", "ctrl+fn+right", "cmd+m", "ctrl+up"
    if sys.platform.startswith("win"):
        return "win+left", "win+right", "win+down", "win+tab"
    return "super+left", "super+right", "super+h", "super+s"


_SNAP_LEFT, _SNAP_RIGHT, _MINIMIZE, _MISSION_CONTROL = _default_hotkeys()


@dataclass(frozen=True)
class InputConfig:
    dry_run: bool = False
    snap_left_keys: str = _SNAP_LEFT
    snap_right_keys: str = _SNAP_RIGHT
    minimize_keys: str = _MINIMIZE
    mission_control_keys: str = _MISSION_CONTROL
    queue_size: int = 512


@dataclass(frozen=True)
class VoiceConfig:
    # "whisper": VAD-segmented speech transcribed by Whisper, triggered when the
    #            utterance starts with a wake phrase. Works for any word.
    # "openwakeword": streaming neural wake word (much lower idle CPU). Needs a
    #            pretrained model name or a custom-trained .onnx path.
    backend: str = "whisper"
    wake_phrases: Tuple[str, ...] = ("computer", "assistant")
    openwakeword_model: str = "hey_jarvis"
    openwakeword_threshold: float = 0.5
    whisper_model: str = "base.en"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    sample_rate: int = 16000
    frame_samples: int = 1280  # 80 ms, the frame size openWakeWord expects
    # Energy VAD: speech starts when RMS exceeds max(min_rms, noise_floor * ratio).
    vad_min_rms: float = 0.012
    vad_noise_ratio: float = 3.0
    vad_start_frames: int = 2
    vad_preroll_frames: int = 4
    silence_end_s: float = 0.7
    min_utterance_s: float = 0.3
    max_utterance_s: float = 8.0
    # After a bare wake word, how long to wait for the command.
    command_timeout_s: float = 5.0
    capture_dir: Path = CAPTURES_DIR


@dataclass(frozen=True)
class AgentConfig:
    """Local OpenAI-compatible chat endpoint. Ollama is the default on this Mac."""

    base_url: str = "http://127.0.0.1:11434/v1"
    model: str = "llama3.1:8b"
    api_key: str = "ollama"
    timeout_s: float = 45.0
    enabled: bool = True


@dataclass(frozen=True)
class AppConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    gesture: GestureConfig = field(default_factory=GestureConfig)
    input: InputConfig = field(default_factory=InputConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    show_debug: bool = True
    enable_voice: bool = True
