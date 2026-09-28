from __future__ import annotations

import pytest

from core.actions import Action, ActionType
from core.agent import clean_transcript, guard_reply, heuristic_actions, is_bare_dictation, parse_agent_response
from core.input_controller import _coalesce_moves, parse_chord
from core.voice_listener import ControlCommand, IntentKind, IntentRouter, split_wake_phrase

WAKE = ("computer", "assistant")


@pytest.mark.parametrize("text,expected", [
    ("Computer, snap left.", "snap left"),
    ("Hey computer, what's on my screen?", "what's on my screen"),
    ("Assistant.", ""),
    ("I bought a new computer yesterday", None),
    ("Thank you.", None),
])
def test_split_wake_phrase(text: str, expected) -> None:
    assert split_wake_phrase(text, WAKE) == expected


def test_router_kinds() -> None:
    router = IntentRouter()
    snap = router.route("snap window left")
    assert snap.kind is IntentKind.MACRO and snap.action == Action(ActionType.SNAP_LEFT)
    pause = router.route("pause gestures")
    assert pause.kind is IntentKind.CONTROL and pause.control is ControlCommand.PAUSE_GESTURES
    assert router.route("summarize this article").kind is IntentKind.AGENT_QUERY
    assert router.route("right click").action == Action(ActionType.RIGHT_CLICK)
    assert router.route("new line").action == Action(ActionType.HOTKEY, keys="enter")
    typing = router.route("take the keyboard")
    assert typing.kind is IntentKind.CONTROL and typing.control is ControlCommand.START_DICTATION
    assert router.route("stop dictation").control is ControlCommand.STOP_DICTATION


def test_parse_agent_response_orders_steps() -> None:
    raw = """```json
    {"steps":[
        {"op":"hotkey","keys":"cmd+space"},
        {"op":"type","text":"Safari"},
        {"op":"key","keys":"enter"},
        {"op":"scroll","amount":0}
    ],"note":"open Safari"}
    ```"""
    actions, note = parse_agent_response(raw)
    assert note == "open Safari"
    assert actions == [
        Action(ActionType.HOTKEY, keys="cmd+space"),
        Action(ActionType.TYPE_TEXT, text="Safari"),
        Action(ActionType.HOTKEY, keys="enter"),
    ]


def test_listening_cue_is_not_a_command() -> None:
    assert clean_transcript("Listening. open Google Chrome") == "open Google Chrome"
    assert clean_transcript("Listening.") == ""
    assert is_bare_dictation("the meeting is at three")
    assert not is_bare_dictation("open Safari")


def test_a_question_is_spoken_instead_of_typed() -> None:
    actions, say = guard_reply(
        "Can you hear me?",
        [Action(ActionType.TYPE_TEXT, text="Can you hear me?")],
        "no action",
        dictation=False,
        editable=False,
    )
    assert actions == []
    assert say == "Yes, I can hear you."


def test_heuristic_types_dictation_and_performs_commands() -> None:
    typed = heuristic_actions("hello there", editable=False, dictation=True)
    assert typed == [Action(ActionType.TYPE_TEXT, text="hello there ")]
    opened = heuristic_actions("open Safari", editable=True, dictation=True)
    assert [a.type for a in opened] == [ActionType.HOTKEY, ActionType.TYPE_TEXT, ActionType.HOTKEY]
    assert opened[1].text == "Safari"
    assert heuristic_actions("see you tomorrow", editable=True, dictation=False)[0].type is ActionType.TYPE_TEXT
    assert heuristic_actions("see you tomorrow", editable=False, dictation=False) == []
    assert heuristic_actions("click the red button", editable=True, dictation=True) == []


def test_coalesce_keeps_order_around_discrete_actions() -> None:
    m = lambda x: Action(ActionType.MOVE_POINTER, x=x)  # noqa: E731
    down, up = Action(ActionType.MOUSE_DOWN), Action(ActionType.MOUSE_UP)
    out = _coalesce_moves([m(0.1), m(0.2), down, m(0.3), m(0.4), up, None])
    assert out == [m(0.2), down, m(0.4), up, None]


def test_parse_chord_aliases() -> None:
    assert parse_chord("Win+Left") == ["cmd", "left"]
    assert parse_chord("control+globe+right") == ["ctrl", "fn", "right"]
    with pytest.raises(ValueError):
        parse_chord("ctrl++")
