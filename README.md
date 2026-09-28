# Gestures

A macOS script that replaces the mouse with two hands and the keyboard with your voice. Start it when you want it. It is not a login item.

The right hand moves the pointer. The left hand clicks, scrolls, and swipes. Speech is typed into a focused text field, or turned into desktop actions when you call the assistant.

## What it does

**Hands.** The camera tracks both hands with MediaPipe. The right hand’s index tip moves the cursor; its pose is ignored, so pointing does not click. The left hand does the rest:

| Left hand | Result |
| --- | --- |
| Pinch | Click. Hold the pinch and move the right hand to drag. |
| Two fingers | Scroll |
| Short fist | Right-click |
| Fist held still | Listen, then the assistant answers out loud |
| Three-finger tap | Double-click |
| Palm swipe left / right | Snap the window left / right |
| Palm swipe down | Minimize |
| Palm swipe up | Mission Control |

**Voice.** Click a text field and the next things you say are typed there. Exact phrases such as “enter”, “backspace”, and “select all” still press keys, including while dictating. Outside a field, say “computer” or “assistant”, or hold a left fist, and a local model (Ollama `llama3.1:8b` by default) decides whether to type, press keys, open an app, or answer aloud.

A short chime means it is listening. Spoken replies use a macOS voice. The microphone is muted while that voice is playing so it does not hear itself.

## Run

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-voice.txt
PYTHONUNBUFFERED=1 .venv/bin/python -u main.py
```

`requirements.txt` is vision and input only. `requirements-voice.txt` adds the microphone, wake word, and Whisper.

```bash
.venv/bin/python -u main.py --headless    # no debug window
.venv/bin/python -u main.py --dry-run -v  # log actions, do not move the mouse
.venv/bin/python -u main.py --no-agent    # gestures and voice without the local model
```

The hand-landmarker model is downloaded into `models/` on first run. That folder is not in this repo.

macOS must allow Camera, Microphone, and Accessibility for the app that runs Python (Terminal, iTerm, Cursor, and so on). Accessibility is what lets the script move the pointer, type, and see whether a text field is focused.

The local assistant expects Ollama at `http://127.0.0.1:11434` with `llama3.1:8b`. Override with `GESTURE_AI_BASE_URL`, `GESTURE_AI_MODEL`, and `GESTURE_AI_API_KEY`, or `--agent-model`.

## Tests

```bash
.venv/bin/python -m pytest tests -q
```
