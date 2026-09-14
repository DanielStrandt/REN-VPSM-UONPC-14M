from __future__ import annotations

"""Gradio demo for the frozen REN VPSM UO NPC 14M release.

The conversation shown in the UI is display-only. Every request is sent to the
model as an independent single-turn query, matching the production UO contract.
"""

import os
import re
import sys
import threading
import time
import hashlib
import urllib.request
from pathlib import Path
from typing import Any

import gradio as gr
import torch

SPACE_ROOT = Path(__file__).resolve().parent
GITHUB_RAW_ROOT = (
    "https://raw.githubusercontent.com/DanielStrandt/REN-VPSM-UONPC-14M/"
    "d8c2f6040388b19077e08e22131b319929aaeb85/"
)
REMOTE_BUNDLE = {
    "runtime/__init__.py": "4669dd5af39c73a44ca52d79fe525f5d7d90af64cf5866f9aa54668c0e536b87",
    "runtime/uo_vpsm_model_ren_v5_scan.py": "f88c67bdb618d2fb2846d8aaf5e5f5c8727589dd803eb1e1ef85da95a7d0f53a",
    "runtime/vpsm_npc_runtime.py": "94f4dc91e3649c06de47e7587224dae2b6b0580b8cefe8c545249a646f9f7a34",
    "tokenizer/tokenizer.json": "2b433ddd27e2d527d7cb3d00a5bad31bf7a2c76f8d42700b8f6935156aff46c6",
    "model/vpsm_v5_uo_npc_final.pt": "4890b25cdc4c1e3c9fdee09d9f298dc0d23e0ff9f80935477d2c36f3f3d80d68",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_bundle() -> Path:
    # When this file is run from the companion GitHub checkout, use its exact
    # files. A standalone HF Space downloads the same pinned release instead.
    local_root = SPACE_ROOT.parent
    if not (local_root / "runtime" / "vpsm_npc_runtime.py").is_file():
        local_root = SPACE_ROOT

    for relative, expected_sha in REMOTE_BUNDLE.items():
        target = local_root / relative
        if target.is_file() and _sha256(target) == expected_sha:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".download")
        try:
            with urllib.request.urlopen(GITHUB_RAW_ROOT + relative, timeout=180) as response:
                with temporary.open("wb") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
            if _sha256(temporary) != expected_sha:
                raise RuntimeError(f"SHA-256 mismatch for {relative}")
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
    return local_root


BUNDLE_ROOT = _ensure_bundle()
sys.path.insert(0, str(BUNDLE_ROOT))

from runtime.vpsm_npc_runtime import (  # noqa: E402
    Persona,
    FinalNPCModel,
    generate_single_turn,
    indefinite_article,
    load_final_model,
    normalize_text,
)


DEFAULT_NAME = "Duncan"
DEFAULT_PROFESSION = "merchant"
DEFAULT_TOWN = "Jhelom"
DEFAULT_TRAIT = "You are proud of your work."
MAX_INPUT_TOKENS = 30
MAX_INPUT_CHARS = 180
MAX_NEW_TOKENS = 18


def _load_model() -> FinalNPCModel:
    requested = os.getenv("VPSM_DEVICE", "").strip()
    device = requested or ("cuda" if torch.cuda.is_available() else "cpu")
    return load_final_model(device=device)


MODEL = _load_model()
MODEL_LOCK = threading.Lock()


def _persona(name: str, profession: str, town: str, trait: str) -> Persona:
    return Persona(
        name=name or DEFAULT_NAME,
        profession=profession or DEFAULT_PROFESSION,
        town=town or DEFAULT_TOWN,
        trait=trait or DEFAULT_TRAIT,
    ).normalized()


def _identity_intent(text: str) -> str:
    normalized = normalize_text(text).lower().strip()
    patterns = {
        "name": (
            "what is your name", "whats your name", "what are you called",
            "what is thy name", "tell me your name",
        ),
        "town": (
            "where are you from", "where do you live", "where is your home",
            "what town are you from", "what is your town", "what is thy home",
        ),
        "trade": (
            "what is your trade", "whats your trade", "what is your profession",
            "what is your job", "what do you do", "what do you do for work",
            "what do you do for a living", "tell me your trade",
        ),
        "full": ("who are you", "describe yourself", "tell me about yourself"),
    }
    for intent, choices in patterns.items():
        if normalized in choices:
            return intent
    return ""


def _identity_reply(persona: Persona, intent: str) -> str:
    p = persona.normalized()
    article = indefinite_article(p.profession)
    if intent == "name":
        return f"my name is {p.name}"
    if intent == "town":
        return f"i am from {p.town}"
    if intent == "trade":
        return f"i am {article} {p.profession}"
    return f"my name is {p.name}\ni am {article} {p.profession} from {p.town}"


def _speech_events(text: str) -> list[str]:
    parts = [re.sub(r"\s+", " ", part).strip() for part in str(text).splitlines()]
    parts = [part for part in parts if part]
    if not parts:
        return ["(no reply)"]
    if len(parts) <= 2:
        return parts
    return [parts[0], " ".join(parts[1:])]


def _status(text: str, tokens: int, elapsed_ms: float, events: list[str]) -> str:
    event_count = len(events)
    return (
        f"**{tokens}** generated model tokens · **{elapsed_ms:.0f} ms** · "
        f"{event_count} speech event{'s' if event_count != 1 else ''}"
    )


def respond(
    player_text: str,
    name: str,
    profession: str,
    town: str,
    trait: str,
    history: list[dict[str, Any]] | None,
):
    raw_input = str(player_text or "").strip()
    if not raw_input:
        return history or [], "", "Enter a short player message.", "", ""

    normalized = normalize_text(raw_input)
    if not normalized:
        return history or [], "", "The message has no usable model text.", "", ""

    input_tokens = len(MODEL.tokenizer.encode(normalized))
    if input_tokens > MAX_INPUT_TOKENS:
        return (
            history or [],
            "",
            f"Message is {input_tokens} model tokens; please keep it at {MAX_INPUT_TOKENS} or fewer.",
            normalized,
            "",
        )

    persona = _persona(name, profession, town, trait)
    started = time.perf_counter()
    intent = _identity_intent(normalized)
    with MODEL_LOCK:
        if intent:
            reply = _identity_reply(persona, intent)
            output_tokens = len(MODEL.tokenizer.encode(reply))
        else:
            reply, output_tokens = generate_single_turn(
                MODEL,
                normalized,
                persona,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.0,
                top_k=0,
            )
    elapsed_ms = (time.perf_counter() - started) * 1000

    events = _speech_events(reply)
    display_reply = "\n".join(f"Speech {i + 1}: {event}" for i, event in enumerate(events))
    updated = list(history or [])
    updated.append({"role": "user", "content": raw_input})
    updated.append({"role": "assistant", "content": display_reply})
    return (
        updated,
        "",
        _status(reply, output_tokens, elapsed_ms, events),
        normalized,
        display_reply,
    )


def clear_chat():
    return [], "", "Ready for a fresh single-turn query.", "", ""


CSS = """
.gradio-container { max-width: 1180px !important; }
.title-card { text-align: center; }
.contract { border-radius: 12px; }
"""


with gr.Blocks(css=CSS, title="REN VPSM UO NPC 14M") as demo:
    gr.Markdown(
        "# REN VPSM UO NPC 14M\n"
        "### A tiny, short-turn villager for Ultima Online",
        elem_classes="title-card",
    )
    gr.Markdown(
        "This demo uses the frozen v7-e14 model release. Each player message is "
        "an independent query; the visible transcript is never sent back to the model.",
        elem_classes="contract",
    )

    with gr.Row():
        with gr.Column(scale=1, min_width=250):
            gr.Markdown("### NPC persona")
            name = gr.Textbox(label="Name", value=DEFAULT_NAME, max_lines=1)
            profession = gr.Textbox(label="Profession", value=DEFAULT_PROFESSION, max_lines=1)
            town = gr.Textbox(label="Town", value=DEFAULT_TOWN, max_lines=1)
            trait = gr.Textbox(label="Trait", value=DEFAULT_TRAIT, lines=2, max_lines=2)
            gr.Markdown(
                "The persona is server-side prompt context only. Game state, quests, "
                "inventory, and permissions remain the UO server's responsibility."
            )

        with gr.Column(scale=2, min_width=420):
            chat = gr.Chatbot(
                label="Display-only conversation",
                height=360,
                placeholder="Your NPC's short replies will appear here.",
            )
            player_text = gr.Textbox(
                label="Player message",
                placeholder="Ask the villager something short...",
                lines=3,
                max_lines=3,
                max_length=MAX_INPUT_CHARS,
            )
            with gr.Row():
                send = gr.Button("Speak", variant="primary")
                clear = gr.Button("Clear")
            status = gr.Markdown("Ready for a fresh single-turn query.")
            normalized_view = gr.Textbox(label="Normalized model input", interactive=False)
            reply_view = gr.Textbox(label="UO speech events", lines=3, interactive=False)

    gr.Markdown(
        f"**Release contract:** input target ≤ {MAX_INPUT_TOKENS} model tokens · "
        f"greedy output cap = {MAX_NEW_TOKENS} model tokens · replies are displayed as "
        "at most two speech events."
    )

    inputs = [player_text, name, profession, town, trait, chat]
    outputs = [chat, player_text, status, normalized_view, reply_view]
    send.click(respond, inputs=inputs, outputs=outputs, concurrency_limit=1)
    player_text.submit(respond, inputs=inputs, outputs=outputs, concurrency_limit=1)
    clear.click(clear_chat, outputs=outputs)


if __name__ == "__main__":
    demo.launch()
