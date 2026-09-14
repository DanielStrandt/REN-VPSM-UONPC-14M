from __future__ import annotations

"""
vpsm_npc_chat_gui.py

Simple no-argument desktop chat app for the frozen VPSM v5 UO NPC final bundle.

Expected location:
    VPSM_UO_NPC_FINAL/apps/vpsm_npc_chat_gui.py

Run:
    python vpsm_npc_chat_gui.py

Production contract:
    * Uses the bundle-relative final checkpoint/tokenizer through runtime.vpsm_npc_runtime.
    * Every player message is an independent single-turn query.
    * The visible transcript is NEVER fed back into the model.
    * The recurrent state is fresh for every query.
    * Greedy generation with an adjustable output cap (default 40 tokens).
    * Identity guard protects name, town, and profession without changing the model.
"""

import queue
import random
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Locate the frozen bundle from this file's expected apps/ location.
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

# Preferred location is bundle/apps/, but running from the bundle root also works.
if (SCRIPT_DIR.parent / "runtime" / "vpsm_npc_runtime.py").is_file():
    BUNDLE_ROOT = SCRIPT_DIR.parent
elif (SCRIPT_DIR / "runtime" / "vpsm_npc_runtime.py").is_file():
    BUNDLE_ROOT = SCRIPT_DIR
else:
    raise SystemExit(
        "Could not find the bundled runtime.\n"
        "Place this file in VPSM_UO_NPC_FINAL/apps/ (preferred) or VPSM_UO_NPC_FINAL/."
    )

sys.path.insert(0, str(BUNDLE_ROOT))

try:
    from runtime.vpsm_npc_runtime import (  # type: ignore
        Persona,
        generate_single_turn,
        indefinite_article,
        load_final_model,
        normalize_text,
    )
except Exception as exc:
    raise SystemExit(f"Could not import bundled VPSM runtime: {exc}") from exc

try:
    import tkinter as tk
    from tkinter import messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError as exc:
    raise SystemExit(
        "Tkinter is required. Standard python.org Windows Python normally includes it."
    ) from exc


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_NAME = "Duncan"
DEFAULT_PROFESSION = "merchant"
DEFAULT_TOWN = "Jhelom"
DEFAULT_TRAIT = "You are proud of your work."

DEFAULT_MAX_NEW_TOKENS = 40
MIN_MAX_NEW_TOKENS = 8
MAX_MAX_NEW_TOKENS = 96
TEMPERATURE = 0.0
TOP_K = 0

KNOWN_TOWNS = {
    "britain",
    "buccaneers den",
    "cove",
    "jhelom",
    "magincia",
    "minoc",
    "moonglow",
    "nujelm",
    "ocllo",
    "serpents hold",
    "skara brae",
    "trinsic",
    "vesper",
    "wind",
    "yew",
}

KNOWN_PROFESSIONS = {
    "alchemist",
    "animal trainer",
    "banker",
    "bard",
    "baker",
    "beekeeper",
    "blacksmith",
    "brewer",
    "butcher",
    "carpenter",
    "carter",
    "cobbler",
    "cook",
    "fisherman",
    "farmer",
    "gardener",
    "guard",
    "healer",
    "herbalist",
    "hunter",
    "innkeeper",
    "jeweler",
    "leathermaker",
    "mage",
    "manager",
    "mason",
    "merchant",
    "miller",
    "miner",
    "potter",
    "provisioner",
    "ropemaker",
    "sailor",
    "scribe",
    "shepherd",
    "shopkeeper",
    "stable hand",
    "tailor",
    "tanner",
    "trader",
    "vintner",
    "weaver",
    "woodcutter",
}

KNOWN_NAMES = {
    "aldric",
    "barnaby",
    "beatrice",
    "bess",
    "bram",
    "cedric",
    "celia",
    "clara",
    "corwin",
    "della",
    "duncan",
    "eamon",
    "edmund",
    "elara",
    "elsbeth",
    "fern",
    "finn",
    "fiona",
    "flora",
    "garrick",
    "george",
    "greta",
    "helena",
    "hester",
    "hilda",
    "hugh",
    "iris",
    "isaac",
    "isadora",
    "ivor",
    "ivy",
    "jasper",
    "joan",
    "jorah",
    "kara",
    "kellan",
    "lark",
    "leof",
    "lydia",
    "lyra",
    "mabel",
    "martha",
    "martin",
    "matthias",
    "mira",
    "morna",
    "myron",
    "ned",
    "nora",
    "olwen",
    "osric",
    "otis",
    "owen",
    "pax",
    "petra",
    "pip",
    "polly",
    "quinn",
    "quintus",
    "raymond",
    "rebecca",
    "rhea",
    "rina",
    "rose",
    "rowan",
    "sage",
    "selena",
    "selma",
    "simon",
    "soren",
    "sybil",
    "tessa",
    "thalia",
    "thorne",
    "tobias",
    "tobin",
    "twilla",
    "ulric",
    "una",
    "valen",
    "vaughn",
    "vera",
    "willa",
    "willard",
    "xander",
    "yorick",
    "ysabel",
    "yvette",
}

TRAITS = (
    "You are proud of your work.",
    "You are practical and plain-spoken.",
    "You are friendly but not overly talkative.",
    "You are patient with strangers.",
    "You are a little gruff but fair.",
    "You are cheerful when work is light.",
    "You are cautious around trouble.",
    "You are kind to all you meet.",
)


# -----------------------------------------------------------------------------
# Identity protection
# -----------------------------------------------------------------------------

@dataclass
class GuardResult:
    text: str
    changed: bool = False
    reason: str = ""
    raw_text: str = ""


def _norm(text: str) -> str:
    return normalize_text(text).replace("\n", " ").strip()


def classify_identity_query(player_text: str) -> str:
    """Return name/town/trade/full for direct identity questions, else ''."""
    t = _norm(player_text)

    full_patterns = (
        r"^who are you$",
        r"^who art thou$",
        r"^describe yourself$",
        r"^tell me about yourself$",
    )
    name_patterns = (
        r"^what is your name$",
        r"^whats your name$",
        r"^what are you called$",
        r"^what art thou called$",
        r"^what is thy name$",
        r"^tell me your name$",
    )
    town_patterns = (
        r"^where are you from$",
        r"^where art thou from$",
        r"^where do you live$",
        r"^where is your home$",
        r"^what town are you from$",
        r"^what is your town$",
        r"^what is thy home$",
    )
    trade_patterns = (
        r"^what is your trade$",
        r"^whats your trade$",
        r"^what is your profession$",
        r"^what is your job$",
        r"^what is thy trade$",
        r"^what do you do$",
        r"^what dost thou do$",
        r"^what do you do for work$",
        r"^what do you do for a living$",
        r"^tell me your trade$",
    )

    for p in full_patterns:
        if re.fullmatch(p, t):
            return "full"
    for p in name_patterns:
        if re.fullmatch(p, t):
            return "name"
    for p in town_patterns:
        if re.fullmatch(p, t):
            return "town"
    for p in trade_patterns:
        if re.fullmatch(p, t):
            return "trade"
    return ""


def canonical_identity_reply(persona: Persona, intent: str) -> str:
    p = persona.normalized()
    name = _norm(p.name)
    town = _norm(p.town)
    profession = _norm(p.profession)
    article = indefinite_article(profession)

    if intent == "name":
        return f"my name is {name}"
    if intent == "town":
        return f"i am from {town}"
    if intent == "trade":
        return f"i am {article} {profession}"
    return f"my name is {name}\ni am {article} {profession} from {town}"


def identity_answer_is_valid(text: str, persona: Persona, intent: str) -> bool:
    t = _norm(text)
    p = persona.normalized()
    name = _norm(p.name)
    town = _norm(p.town)
    profession = _norm(p.profession)

    if not t:
        return False
    if intent == "name":
        return name in t
    if intent == "town":
        return town in t
    if intent == "trade":
        return profession in t
    if intent == "full":
        return name in t and town in t and profession in t
    return True


def repair_explicit_identity_claims(text: str, persona: Persona) -> tuple[str, list[str]]:
    """Narrowly repair explicit self-identity drift in otherwise ordinary replies."""
    if not text.strip():
        return text, []

    p = persona.normalized()
    name = _norm(p.name)
    town = _norm(p.town)
    profession = _norm(p.profession)
    article = indefinite_article(profession)
    out = text.strip()
    reasons: list[str] = []

    # Explicit naming phrases are identity claims, so canonicalizing them is safe.
    name_claims = (
        (r"\bmy name is\s+[a-z]+(?:\s+[a-z]+)?", f"my name is {name}"),
        (r"\bi am called\s+[a-z]+(?:\s+[a-z]+)?", f"i am called {name}"),
        (r"\bthey call me\s+[a-z]+(?:\s+[a-z]+)?", f"they call me {name}"),
    )
    for pattern, replacement in name_claims:
        new = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        if new != out:
            out = new
            reasons.append("name claim")

    wrong_names = sorted(KNOWN_NAMES - {name}, key=len, reverse=True)
    if wrong_names:
        alt = "|".join(re.escape(v) for v in wrong_names)
        new = re.sub(
            rf"\bi am\s+(?:{alt})\b",
            f"i am {name}",
            out,
            flags=re.IGNORECASE,
        )
        if new != out:
            out = new
            reasons.append("wrong name")

    wrong_towns = sorted(KNOWN_TOWNS - {town}, key=len, reverse=True)
    if wrong_towns:
        alt = "|".join(re.escape(v) for v in wrong_towns)
        # Only first-person origin claims; third-party place mentions are untouched.
        pattern = rf"(\bi am\b[^\n.!?]{{0,48}}\bfrom\s+)(?:{alt})\b"
        new = re.sub(pattern, lambda m: m.group(1) + town, out, flags=re.IGNORECASE)
        if new != out:
            out = new
            reasons.append("wrong town")

    wrong_professions = sorted(KNOWN_PROFESSIONS - {profession}, key=len, reverse=True)
    if wrong_professions:
        alt = "|".join(re.escape(v) for v in wrong_professions)
        patterns = (
            (rf"\bi am\s+(?:a|an)\s+(?:{alt})\b", f"i am {article} {profession}"),
            (rf"\bi work as\s+(?:a|an)?\s*(?:{alt})\b", f"i work as {article} {profession}"),
            (rf"\bmy trade is\s+(?:a|an)?\s*(?:{alt})\b", f"my trade is {profession}"),
        )
        for pattern, replacement in patterns:
            new = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
            if new != out:
                out = new
                reasons.append("wrong profession")

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out).strip()
    return out, reasons


def protect_identity(player_text: str, raw_text: str, persona: Persona, enabled: bool) -> GuardResult:
    raw = (raw_text or "").strip()
    if not enabled:
        return GuardResult(text=raw, raw_text=raw)

    intent = classify_identity_query(player_text)
    repaired, reasons = repair_explicit_identity_claims(raw, persona)

    if intent and not identity_answer_is_valid(repaired, persona, intent):
        fixed = canonical_identity_reply(persona, intent)
        return GuardResult(
            text=fixed,
            changed=fixed != raw,
            reason=f"direct {intent} identity correction",
            raw_text=raw,
        )

    if reasons:
        return GuardResult(
            text=repaired,
            changed=repaired != raw,
            reason=", ".join(dict.fromkeys(reasons)),
            raw_text=raw,
        )

    return GuardResult(text=raw, raw_text=raw)


# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------

class NPCChatApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("VPSM v5 UO NPC - Final Chat")
        self.root.geometry("1050x760")
        self.root.minsize(860, 620)

        self.loaded = None
        self.busy = False
        self.events: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.identity_fix_count = 0
        self.turn_count = 0
        self.last_raw = ""

        self._build_ui()
        self.root.after(75, self._poll_events)
        self._load_model_async()

    def _build_ui(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Muted.TLabel", font=("Segoe UI", 9))
        style.configure("Send.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="VPSM v5 UO NPC Final", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Loading final model...")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        body = ttk.Panedwindow(outer, orient="horizontal")
        body.pack(fill="both", expand=True)

        left = ttk.Frame(body, padding=(0, 0, 8, 0))
        right = ttk.Frame(body, padding=(8, 0, 0, 0))
        body.add(left, weight=4)
        body.add(right, weight=2)

        self.chat = ScrolledText(
            left,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 11),
            padx=10,
            pady=10,
        )
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_configure("you", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("npc", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("system", font=("Segoe UI", 9, "italic"))
        self.chat.tag_configure("body", spacing3=9)

        input_row = ttk.Frame(left)
        input_row.pack(fill="x", pady=(8, 0))
        self.input_box = tk.Text(input_row, height=3, wrap="word", font=("Segoe UI", 11))
        self.input_box.pack(side="left", fill="x", expand=True)
        self.input_box.bind("<Return>", self._return_pressed)
        self.input_box.bind("<Shift-Return>", self._shift_return_pressed)

        self.send_button = ttk.Button(
            input_row,
            text="Send",
            style="Send.TButton",
            command=self.send,
            state="disabled",
        )
        self.send_button.pack(side="left", padx=(8, 0), fill="y")

        persona_box = ttk.LabelFrame(right, text="NPC persona", padding=8)
        persona_box.pack(fill="x")

        self.name_var = tk.StringVar(value=DEFAULT_NAME)
        self.prof_var = tk.StringVar(value=DEFAULT_PROFESSION)
        self.town_var = tk.StringVar(value=DEFAULT_TOWN)
        self.trait_var = tk.StringVar(value=DEFAULT_TRAIT)

        self._entry(persona_box, "Name", self.name_var, 0)
        self._entry(persona_box, "Profession", self.prof_var, 1)
        self._entry(persona_box, "Town", self.town_var, 2)
        self._entry(persona_box, "Trait", self.trait_var, 3)

        random_row = ttk.Frame(persona_box)
        random_row.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(random_row, text="Random villager", command=self.randomize_persona).pack(side="left")
        ttk.Label(random_row, text="Changes apply to next message.", style="Muted.TLabel").pack(
            side="left", padx=(8, 0)
        )

        guard_box = ttk.LabelFrame(right, text="Identity protection", padding=8)
        guard_box.pack(fill="x", pady=(8, 0))
        self.guard_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            guard_box,
            text="Protect name / town / profession",
            variable=self.guard_var,
        ).pack(anchor="w")
        self.guard_status_var = tk.StringVar(value="Corrections: 0")
        ttk.Label(guard_box, textvariable=self.guard_status_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(
            guard_box,
            text="Only explicit identity mistakes are corrected after generation.",
            wraplength=290,
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(3, 0))

        mode_box = ttk.LabelFrame(right, text="Inference mode", padding=8)
        mode_box.pack(fill="x", pady=(8, 0))
        ttk.Label(
            mode_box,
            text="SINGLE TURN / FRESH STATE",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            mode_box,
            text=(
                "The transcript is display-only. Every Send uses only the current persona "
                "and current player message. Greedy generation stops naturally at EOS; "
                "the token setting below is only a hard safety cap."
            ),
            wraplength=290,
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 0))

        cap_row = ttk.Frame(mode_box)
        cap_row.pack(fill="x", pady=(7, 0))
        ttk.Label(cap_row, text="Max output tokens").pack(side="left")
        self.max_tokens_var = tk.IntVar(value=DEFAULT_MAX_NEW_TOKENS)
        ttk.Spinbox(
            cap_row,
            textvariable=self.max_tokens_var,
            from_=MIN_MAX_NEW_TOKENS,
            to=MAX_MAX_NEW_TOKENS,
            increment=4,
            width=7,
        ).pack(side="right")

        prompt_box = ttk.LabelFrame(right, text="Current system prompt", padding=8)
        prompt_box.pack(fill="x", pady=(8, 0))
        self.prompt_preview = tk.Text(prompt_box, height=5, wrap="word", font=("Consolas", 9))
        self.prompt_preview.pack(fill="x")
        self.prompt_preview.configure(state="disabled")

        for var in (self.name_var, self.prof_var, self.town_var, self.trait_var):
            var.trace_add("write", lambda *_: self._refresh_prompt())
        self._refresh_prompt()

        debug_box = ttk.LabelFrame(right, text="Last identity correction", padding=8)
        debug_box.pack(fill="both", expand=True, pady=(8, 0))
        self.guard_detail = ScrolledText(
            debug_box,
            height=7,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
        )
        self.guard_detail.pack(fill="both", expand=True)

        bottom = ttk.Frame(right)
        bottom.pack(fill="x", pady=(8, 0))
        ttk.Button(bottom, text="Clear chat", command=self.clear_chat).pack(side="left")
        ttk.Button(bottom, text="Reload model", command=self._load_model_async).pack(side="right")

        self.input_box.focus_set()

    def _entry(self, parent, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=2)
        parent.columnconfigure(1, weight=1)

    def persona(self) -> Persona:
        return Persona(
            name=self.name_var.get(),
            profession=self.prof_var.get(),
            town=self.town_var.get(),
            trait=self.trait_var.get(),
        ).normalized()

    def _refresh_prompt(self) -> None:
        if not hasattr(self, "prompt_preview"):
            return
        text = self.persona().system_prompt()
        self.prompt_preview.configure(state="normal")
        self.prompt_preview.delete("1.0", "end")
        self.prompt_preview.insert("1.0", text)
        self.prompt_preview.configure(state="disabled")

    def randomize_persona(self) -> None:
        if self.busy:
            return
        self.name_var.set(random.choice(sorted(KNOWN_NAMES)).title())
        self.prof_var.set(random.choice(sorted(KNOWN_PROFESSIONS)))
        self.town_var.set(random.choice(sorted(KNOWN_TOWNS)).title())
        self.trait_var.set(random.choice(TRAITS))

    def _load_model_async(self) -> None:
        if self.busy:
            return
        self.busy = True
        self.send_button.configure(state="disabled")
        self.status_var.set("Loading final model...")

        def worker() -> None:
            try:
                loaded = load_final_model()
                self.events.put(("loaded", loaded))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "loaded":
                    self.loaded = payload
                    self.busy = False
                    self.send_button.configure(state="normal")
                    dev = str(self.loaded.device)
                    amp = (
                        str(self.loaded.amp_dtype).replace("torch.", "")
                        if self.loaded.amp_dtype is not None
                        else "fp32"
                    )
                    self.status_var.set(
                        f"Ready | {dev} | {amp} | epoch {self.loaded.checkpoint_epoch}"
                    )
                    self._append_system(
                        f"Final model loaded: {self.loaded.parameter_count:,} parameters. "
                        "Single-turn mode is active."
                    )
                elif kind == "response":
                    self._finish_response(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self.busy = False
                    self.send_button.configure(state="normal" if self.loaded is not None else "disabled")
                    self.status_var.set("Error")
                    messagebox.showerror("VPSM error", str(payload))
        except queue.Empty:
            pass
        self.root.after(75, self._poll_events)

    def _return_pressed(self, event):
        # Return sends. Shift+Return inserts a newline.
        if event.state & 0x0001:
            return None
        self.send()
        return "break"

    def _shift_return_pressed(self, event):
        self.input_box.insert("insert", "\n")
        return "break"

    def send(self) -> None:
        if self.busy or self.loaded is None:
            return

        player_text = self.input_box.get("1.0", "end").strip()
        if not player_text:
            return

        self.input_box.delete("1.0", "end")
        p = self.persona()
        guard_enabled = bool(self.guard_var.get())
        try:
            max_new_tokens = int(self.max_tokens_var.get())
        except (TypeError, ValueError, tk.TclError):
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS
        max_new_tokens = max(MIN_MAX_NEW_TOKENS, min(MAX_MAX_NEW_TOKENS, max_new_tokens))
        self.max_tokens_var.set(max_new_tokens)

        self._append_chat("You", player_text, "you")
        self.busy = True
        self.send_button.configure(state="disabled")
        self.status_var.set("Thinking...")
        started = time.perf_counter()

        def worker() -> None:
            try:
                # generate_single_turn() is deliberately stateless across player turns.
                raw, token_count = generate_single_turn(
                    self.loaded,
                    player_text,
                    p,
                    max_new_tokens=max_new_tokens,
                    temperature=TEMPERATURE,
                    top_k=TOP_K,
                )
                guarded = protect_identity(player_text, raw, p, guard_enabled)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.events.put(
                    (
                        "response",
                        {
                            "player": player_text,
                            "persona": p,
                            "guard": guarded,
                            "tokens": token_count,
                            "max_new_tokens": max_new_tokens,
                            "elapsed_ms": elapsed_ms,
                        },
                    )
                )
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_response(self, result: dict) -> None:
        guard: GuardResult = result["guard"]
        p: Persona = result["persona"]
        response = guard.text.strip() or "[no response]"

        self.turn_count += 1
        self._append_chat(p.name, response, "npc")

        if guard.changed:
            self.identity_fix_count += 1
            self.guard_status_var.set(f"Corrections: {self.identity_fix_count}")
            self._show_guard_detail(guard)

        self.busy = False
        self.send_button.configure(state="normal")
        token_count = int(result["tokens"])
        hard_cap = int(result.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
        cap_note = " | HARD CAP HIT" if token_count >= hard_cap else ""
        self.status_var.set(
            f"Ready | {result['elapsed_ms']:.0f} ms | {token_count} generated tokens{cap_note}"
        )
        self.input_box.focus_set()

    def _show_guard_detail(self, guard: GuardResult) -> None:
        text = (
            f"Reason: {guard.reason}\n\n"
            f"Raw model output:\n{guard.raw_text or '[empty]'}\n\n"
            f"Displayed output:\n{guard.text or '[empty]'}"
        )
        self.guard_detail.configure(state="normal")
        self.guard_detail.delete("1.0", "end")
        self.guard_detail.insert("1.0", text)
        self.guard_detail.configure(state="disabled")

    def _append_chat(self, speaker: str, text: str, label_tag: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}\n", label_tag)
        self.chat.insert("end", text.strip() + "\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _append_system(self, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", text.strip() + "\n\n", "system")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def clear_chat(self) -> None:
        if self.busy:
            return
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.guard_detail.configure(state="normal")
        self.guard_detail.delete("1.0", "end")
        self.guard_detail.configure(state="disabled")
        self.turn_count = 0
        self.input_box.focus_set()


def main() -> None:
    root = tk.Tk()
    NPCChatApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
