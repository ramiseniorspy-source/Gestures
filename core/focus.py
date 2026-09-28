"""Which control is focused, so speech can be typed or treated as a command.

Uses the macOS accessibility tree. The same permission that lets this script
move the pointer also lets it read the focused role. Failures return an empty
focus rather than blocking voice or gestures.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

logger = logging.getLogger(__name__)

EDITABLE_ROLES = frozenset({
    "AXTextField",
    "AXTextArea",
    "AXComboBox",
    "AXSearchField",
    "AXTextView",
    "AXSecureTextField",
})


@dataclass(frozen=True)
class FocusInfo:
    app_name: str = ""
    role: str = ""
    editable: bool = False


def _attribute(element: object, name: str) -> str:
    import ApplicationServices as AS

    err, value = AS.AXUIElementCopyAttributeValue(element, name, None)
    if err != 0 or value is None:
        return ""
    return str(value)


def read_focus() -> FocusInfo:
    if sys.platform != "darwin":
        return FocusInfo()
    try:
        import ApplicationServices as AS

        system = AS.AXUIElementCreateSystemWide()
        err, focused = AS.AXUIElementCopyAttributeValue(system, "AXFocusedUIElement", None)
        if err != 0 or focused is None:
            return FocusInfo()
        role = _attribute(focused, "AXRole")
        description = _attribute(focused, "AXRoleDescription").lower()
        app_name = ""
        err, app = AS.AXUIElementCopyAttributeValue(system, "AXFocusedApplication", None)
        if err == 0 and app is not None:
            app_name = _attribute(app, "AXTitle")
        editable = role != "AXSecureTextField" and (
            role in EDITABLE_ROLES
            or any(token in description for token in ("text field", "text area", "search field", "text entry"))
        )
        return FocusInfo(app_name=app_name, role=role, editable=editable)
    except Exception:
        logger.debug("Focus lookup failed", exc_info=True)
        return FocusInfo()
