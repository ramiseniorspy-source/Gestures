"""Decide whether a spoken phrase should be typed or performed.

The local model (Ollama by default) sees the phrase and what is focused. It
returns an ordered list of keyboard and mouse steps. If the model is off or
unreachable, a small heuristic does the same job: type into a text field,
carry out an obvious command, and leave everything else alone.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import urllib.error
import urllib.request
from typing import List, Optional, Tuple

from core.actions import Action, ActionType
from core.config import AgentConfig
from core.focus import FocusInfo

logger = logging.getLogger(__name__)

_MOD = "cmd" if sys.platform == "darwin" else "ctrl"
_MAX_STEPS = 8

_SYSTEM = (
    "You decide how a spoken phrase should control a Mac. "
    "Reply with JSON only, no markdown, using this shape: "
    '{"steps":[{"op":"type","text":"..."},{"op":"hotkey","keys":"cmd+c"},'
    '{"op":"key","keys":"enter"},{"op":"click"},{"op":"right_click"},'
    '{"op":"double_click"},{"op":"scroll","amount":4},'
    '{"op":"snap_left"},{"op":"snap_right"},{"op":"minimize"},'
    '{"op":"mission_control"}],"say":"short spoken reply"}. '
    "Put steps in the order they should happen. "
    "type inserts literal text at the caret. "
    "hotkey is a chord such as cmd+t or ctrl+up. key is one of "
    "enter, tab, backspace, delete, esc, space, up, down, left, right. "
    "scroll amount is positive to scroll up and negative to scroll down. "
    "In dictation mode, type their words, cleaned up, unless they are clearly "
    "telling the computer to do something. Do that even if the focused role is unknown. "
    "If they are talking to you, answer the actual question in say with one or two "
    "natural sentences and leave steps empty. Never type that answer. "
    "Never reply with only 'listening' or 'I'm listening'. "
    "When you perform an action, say is one short spoken confirmation of what you did. "
    "To open or switch to an app, use hotkey cmd+space, then type the app name, "
    "then key enter. "
    "Click, right-click, or double-click only when they ask for that click and "
    "do not name a target you cannot see. "
    "If nothing should happen, return an empty steps list. "
    "Never invent shell commands or mouse coordinates."
)

def _normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


_COMMAND_START = re.compile(
    r"^(open|click|right click|double click|scroll|press|hit|launch|quit|close|"
    r"switch|search|go to|select|delete|copy|paste|undo|cut|snap|minimize|"
    r"mission|show|save|new tab|new line|backspace|enter)\b"
)


def _loads_object(raw: str) -> dict:
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("agent response was not a JSON object")
    return value


def _step_action(step: object) -> Optional[Action]:
    if not isinstance(step, dict):
        return None
    op = str(step.get("op", "")).strip().lower()
    if op == "type":
        text = str(step.get("text", ""))
        return Action(ActionType.TYPE_TEXT, text=text) if text else None
    if op in ("hotkey", "key"):
        keys = str(step.get("keys", "")).strip().lower()
        return Action(ActionType.HOTKEY, keys=keys) if keys else None
    if op == "click":
        return Action(ActionType.CLICK)
    if op == "right_click":
        return Action(ActionType.RIGHT_CLICK)
    if op == "double_click":
        return Action(ActionType.DOUBLE_CLICK)
    if op == "scroll":
        try:
            amount = int(step.get("amount", 0))
        except (TypeError, ValueError):
            return None
        return Action(ActionType.SCROLL, amount=amount) if amount else None
    if op == "snap_left":
        return Action(ActionType.SNAP_LEFT)
    if op == "snap_right":
        return Action(ActionType.SNAP_RIGHT)
    if op == "minimize":
        return Action(ActionType.MINIMIZE)
    if op == "mission_control":
        return Action(ActionType.MISSION_CONTROL)
    return None


def parse_agent_response(raw: str) -> Tuple[List[Action], str]:
    """Turn model JSON into actions. Raises ValueError on unusable output."""
    payload = _loads_object(raw)
    note = str(payload.get("note", "")).strip()
    say = str(payload.get("say", "")).strip()
    if note.lower() in ("no action", "none", "n/a"):
        note = ""
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        raise ValueError("steps was not a list")
    actions: List[Action] = []
    for step in steps[:_MAX_STEPS]:
        action = _step_action(step)
        if action is not None:
            actions.append(action)
    return actions, say or note


def dictation_text(transcript: str) -> str:
    words = " ".join(transcript.strip().split())
    if not words:
        return ""
    return words if words.endswith((" ", "\n")) else words + " "


def _typed_payload(query: str) -> Optional[str]:
    match = re.match(r"(?i)^(type|write)\s+(.+)$", query.strip())
    if match is None:
        return None
    return dictation_text(match.group(2))


def _open_target(query: str) -> Optional[str]:
    match = re.match(r"(?i)^open\s+(.+?)\s*$", query.strip().rstrip("."))
    if match is None:
        return None
    name = match.group(1).strip()
    return name or None


def _simple_command(text: str) -> Optional[List[Action]]:
    if text in ("scroll up",):
        return [Action(ActionType.SCROLL, amount=5)]
    if text in ("scroll down",):
        return [Action(ActionType.SCROLL, amount=-5)]
    if text in ("right click",):
        return [Action(ActionType.RIGHT_CLICK)]
    if text in ("double click",):
        return [Action(ActionType.DOUBLE_CLICK)]
    if text in ("enter", "press enter", "hit enter", "new line", "newline"):
        return [Action(ActionType.HOTKEY, keys="enter")]
    if text in ("backspace", "delete", "delete that"):
        return [Action(ActionType.HOTKEY, keys="backspace")]
    if text in ("tab", "press tab"):
        return [Action(ActionType.HOTKEY, keys="tab")]
    if text in ("select all",):
        return [Action(ActionType.HOTKEY, keys=f"{_MOD}+a")]
    if text in ("cut",):
        return [Action(ActionType.HOTKEY, keys=f"{_MOD}+x")]
    if text in ("close window", "close the window", "close tab", "close the tab"):
        return [Action(ActionType.HOTKEY, keys=f"{_MOD}+w")]
    if text in ("new tab",):
        return [Action(ActionType.HOTKEY, keys=f"{_MOD}+t")]
    if text in ("switch app", "switch application", "switch window", "switch apps"):
        return [Action(ActionType.HOTKEY, keys=f"{_MOD}+tab")]
    if text in ("mission control",):
        return [Action(ActionType.MISSION_CONTROL)]
    if text in ("save", "save file", "save the file", "save document"):
        return [Action(ActionType.HOTKEY, keys=f"{_MOD}+s")]
    return None


_ECHO = re.compile(r"^(?:i am |i'm |im )?listening\b[.,!\s]*", re.IGNORECASE)
_USELESS_SAY = {
    "", "listening", "i m listening", "im listening", "i am listening",
    "no action", "none", "n a", "done", "opening", "ok", "okay",
}


def clean_transcript(text: str) -> str:
    """Drop the assistant's own 'listening' cue if the mic heard it."""
    return _ECHO.sub("", text).strip(" .,")


def is_bare_dictation(query: str) -> bool:
    """Prose to type now, rather than a command for the model to interpret."""
    text = _normalize(clean_transcript(query))
    if not text or _COMMAND_START.search(text):
        return False
    if _simple_command(text) is not None or _open_target(query) is not None:
        return False
    return True


def conversational_reply(query: str) -> str:
    text = _normalize(query)
    if "hear" in text:
        return "Yes, I can hear you."
    if text in ("hello", "hi", "hey") or text.startswith(("hello ", "hi ", "hey ")):
        return "Hello. What should I do?"
    if "?" in query or text.startswith(("what", "who", "why", "how", "when", "where", "can you", "could you")):
        return "I heard you. Tell me a command, like open Safari, or say take the keyboard to type."
    return ""


def guard_reply(
    query: str,
    actions: List[Action],
    say: str,
    *,
    dictation: bool,
    editable: bool,
) -> Tuple[List[Action], str]:
    """Questions are spoken. Typing a chat reply into a random window is dropped."""
    if _normalize(say) in _USELESS_SAY:
        say = ""
    if dictation:
        return actions, ""
    query_norm = _normalize(clean_transcript(query))
    if actions and any(action.type is not ActionType.TYPE_TEXT for action in actions):
        actions = [
            action for action in actions
            if not (action.type is ActionType.TYPE_TEXT and _normalize(action.text) == query_norm)
        ]
    if actions and not editable and all(action.type is ActionType.TYPE_TEXT for action in actions):
        actions = []
    if not say:
        say = conversational_reply(query)
    if not say and actions:
        opened = next((action.text for action in actions if action.type is ActionType.TYPE_TEXT), "")
        say = f"Opening {opened}." if opened else "Done."
    if not say and not actions:
        say = "I heard you, but I don't know what to do with that."
    return actions, say


def heuristic_actions(query: str, *, editable: bool, dictation: bool) -> List[Action]:
    """Offline stand-in for the model. Used when the model is disabled or down."""
    text = _normalize(query)
    typed = _typed_payload(query)
    if typed is not None:
        return [Action(ActionType.TYPE_TEXT, text=typed)] if typed else []
    simple = _simple_command(text)
    if simple is not None:
        return simple
    app = _open_target(query)
    if app is not None:
        return [
            Action(ActionType.HOTKEY, keys=f"{_MOD}+space"),
            Action(ActionType.TYPE_TEXT, text=app),
            Action(ActionType.HOTKEY, keys="enter"),
        ]
    if (dictation or editable) and not _COMMAND_START.search(text):
        words = dictation_text(query)
        return [Action(ActionType.TYPE_TEXT, text=words)] if words else []
    return []


class DesktopAgent:
    def __init__(self, config: AgentConfig) -> None:
        self._config = config

    def decide(self, query: str, focus: FocusInfo, *, dictation: bool) -> Tuple[List[Action], str]:
        if not query.strip():
            return [], ""
        if not self._config.enabled:
            actions = heuristic_actions(query, editable=focus.editable, dictation=dictation)
            return guard_reply(query, actions, "", dictation=dictation, editable=focus.editable)
        try:
            raw = self._complete(query, focus, dictation=dictation)
            actions, say = parse_agent_response(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError, KeyError, OSError) as exc:
            logger.warning("Agent model unavailable (%s); using the offline decider", exc)
            actions = heuristic_actions(query, editable=focus.editable, dictation=dictation)
            say = ""
        return guard_reply(query, actions, say, dictation=dictation, editable=focus.editable)

    def _complete(self, query: str, focus: FocusInfo, *, dictation: bool) -> str:
        user = (
            f"mode: {'dictation' if dictation else 'command'}\n"
            f"focused app: {focus.app_name or 'unknown'}\n"
            f"focused role: {focus.role or 'unknown'}\n"
            f"editable: {str(focus.editable).lower()}\n"
            f"phrase: {query}"
        )
        payload = {
            "model": self._config.model,
            "temperature": 0.1,
            "stream": False,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            self._config.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._config.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self._config.timeout_s) as response:
            body = json.loads(response.read().decode())
        content = body["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("agent model returned an empty message")
        return content
