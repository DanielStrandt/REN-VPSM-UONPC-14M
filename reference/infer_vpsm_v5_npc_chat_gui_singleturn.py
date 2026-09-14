from __future__ import annotations

"""
infer_vpsm_v5_npc_chat_gui_singleturn.py

Desktop chat GUI for the VPSM v5 Ultima Online villager model.

Primary deployment target:
    F:\\uovpsm\\vpsm_v5_npc_sft_chat_v7_semantic_routing_final\\checkpoints\\epoch_14.pt

Features
--------
* Tkinter desktop GUI; no extra GUI package required.
* Uses the frozen 5,000-token tokenizer and unchanged VPSM v5 model.
* Uses the same prompt format and greedy/sampling generation path as training.
* Editable NPC persona: name, profession, town, trait.
* Turn-isolated chat: every Send is exactly one system + one user query.
* The visible GUI transcript is never fed back into the model.
* Identity protection layer:
    - validates direct name/town/trade/self questions;
    - replaces a bad direct identity answer with the canonical persona fact;
    - repairs explicit self-claims that drift to a different town/profession/name;
    - repairs identity drift only in the displayed response.
* Guard intervention log showing raw model output when a correction occurs.
* Save transcript as JSON.

The protection layer is deliberately narrow. It does not rewrite ordinary chat,
lore, profession knowledge, or semantic mistakes. It protects only the NPC's
identity facts.

Requirements
------------
    pip install torch tokenizers

The following model source must be importable (normally beside this script):
    uo_vpsm_model_ren_v5_scan.py

Typical use
-----------
    python infer_vpsm_v5_npc_chat_gui_singleturn.py

Optional example
----------------
    python infer_vpsm_v5_npc_chat_gui_singleturn.py --name Duncan --profession merchant --town Jhelom
"""

import argparse
import contextlib
import hashlib
import inspect
import json
import os
import queue
import random
import re
import string
import sys
import threading
import time
import traceback
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

try:
    import torch
except ImportError as exc:
    raise SystemExit("Missing PyTorch. Install the CUDA build used by the project.") from exc

try:
    from tokenizers import Tokenizer
except ImportError as exc:
    raise SystemExit("Missing tokenizers. Install with: pip install tokenizers") from exc

try:
    from uo_vpsm_model_ren_v5_scan import UOVPSMConfig, UOVPSMModel
except ImportError as exc:
    raise SystemExit(
        "Could not import uo_vpsm_model_ren_v5_scan.py. Put this GUI script in "
        "F:\\uovpsm beside the model source, or add that folder to PYTHONPATH."
    ) from exc

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
except ImportError as exc:
    raise SystemExit("Tkinter is required. Standard python.org Windows Python includes it.") from exc


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_CHECKPOINT = Path(
    r"F:\uovpsm\vpsm_v5_npc_sft_chat_v7_semantic_routing_final\checkpoints\epoch_14.pt"
)
DEFAULT_TOKENIZER = Path(
    r"F:\uovpsm\vpsm_v5_scan_tinystories\tokenizer\tokenizer.json"
)
DEFAULT_NAME = "Duncan"
DEFAULT_PROFESSION = "merchant"
DEFAULT_TOWN = "Jhelom"
DEFAULT_TRAIT = "You are proud of your work."

VOCAB_SIZE = 5000
NORMALIZATION_VERSION = "tiny-english-v2-implicit-boundary"

SPECIAL_TOKENS: Tuple[str, ...] = (
    "<pad>", "<bos>", "<eos>", "<join>", "<sent>", "<unk>",
)
LEXICAL_RE = re.compile(r"^[a-z]+$")
MULTISPACE_RE = re.compile(r"[ \t\f\v]+")
MULTINEWLINE_RE = re.compile(r" *\n+ *")
NON_MODEL_CHAR_RE = re.compile(r"[^a-z \n]")
INTEGER_RE = re.compile(r"\d+")
DECIMAL_RE = re.compile(r"(?<!\w)(\d+)\.(\d+)(?!\w)")

PUNCT_TRANSLATION = {ord(ch): " " for ch in string.punctuation}
PUNCT_TRANSLATION[ord(".")] = "\n"
PUNCT_TRANSLATION[ord("!")] = "\n"
PUNCT_TRANSLATION[ord("?")] = "\n"
PUNCT_TRANSLATION[ord("'")] = None
PUNCT_TRANSLATION[ord("`")] = None

UNICODE_EXPANSIONS = {
    "\u00e6": "ae", "\u0153": "oe", "\u00df": "ss", "\u00f8": "o",
    "\u0142": "l", "\u00f0": "d", "\u00fe": "th",
}
DIGIT_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
}
ONES = (
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
)
TENS = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)

ROLE_CANONICAL = {
    "system": "system",
    "developer": "system",
    "user": "user",
    "human": "user",
    "player": "user",
    "assistant": "assistant",
    "bot": "assistant",
    "npc": "assistant",
}
PROMPT_CUES = {
    "user": "the player says",
    "assistant": "the character says",
}
NAME_RE = re.compile(r"\byour\s+name\s+is\s+([^.!?\n;:]+)", re.IGNORECASE)

# Lists are only used by the post-generation identity guard. They do not constrain
# the persona editor: a custom profession/town is always allowed.
KNOWN_TOWNS = {
    "britain", "buccaneers den", "cove", "jhelom", "magincia", "minoc",
    "moonglow", "nujelm", "ocllo", "serpents hold", "skara brae", "trinsic",
    "vesper", "wind", "yew",
}
KNOWN_PROFESSIONS = {
    "alchemist", "animal trainer", "banker", "bard", "baker", "beekeeper",
    "blacksmith", "brewer", "butcher", "carpenter", "carter", "cobbler",
    "cook", "fisherman", "farmer", "gardener", "guard", "healer", "herbalist",
    "hunter", "innkeeper", "jeweler", "leathermaker", "mage", "manager",
    "mason", "merchant", "miller", "miner", "potter", "provisioner",
    "ropemaker", "sailor", "scribe", "shepherd", "shopkeeper", "stable hand",
    "tailor", "tanner", "trader", "vintner", "weaver", "woodcutter",
}
KNOWN_NAMES = {
    "aldric", "barnaby", "beatrice", "bess", "bram", "cedric", "celia",
    "clara", "corwin", "della", "duncan", "eamon", "edmund", "elara",
    "elsbeth", "fern", "finn", "fiona", "flora", "garrick", "george",
    "greta", "helena", "hester", "hilda", "hugh", "iris", "isaac", "isadora",
    "ivor", "ivy", "jasper", "joan", "jorah", "kara", "kellan", "lark",
    "leof", "lydia", "lyra", "mabel", "martha", "martin", "matthias", "mira",
    "morna", "myron", "ned", "nora", "olwen", "osric", "otis", "owen", "pax",
    "petra", "pip", "polly", "quinn", "quintus", "raymond", "rebecca", "rhea",
    "rina", "rose", "rowan", "sage", "selena", "selma", "simon", "soren",
    "sybil", "tessa", "thalia", "thorne", "tobias", "tobin", "twilla", "ulric",
    "una", "valen", "vaughn", "vera", "willa", "willard", "xander", "yorick",
    "ysabel", "yvette",
}


# =============================================================================
# Frozen tokenizer implementation - identical behavior to v7 training
# =============================================================================


def _under_thousand(n: int) -> str:
    out: List[str] = []
    if n >= 100:
        out.extend((ONES[n // 100], "hundred"))
        n %= 100
    if n >= 20:
        out.append(TENS[n // 10])
        if n % 10:
            out.append(ONES[n % 10])
    elif n > 0:
        out.append(ONES[n])
    return " ".join(out) if out else "zero"


def integer_to_words(raw: str) -> str:
    if len(raw) > 9:
        return " ".join(DIGIT_WORD[d] for d in raw)
    n = int(raw)
    if n == 0:
        return "zero"
    if n < 1000:
        return _under_thousand(n)
    parts: List[str] = []
    for scale, name in (
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1000, "thousand"),
    ):
        if n >= scale:
            q, n = divmod(n, scale)
            parts.extend((_under_thousand(q), name))
    if n:
        parts.append(_under_thousand(n))
    return " ".join(parts)


def _decimal_replacement(match: re.Match[str]) -> str:
    return (
        integer_to_words(match.group(1))
        + " point "
        + " ".join(DIGIT_WORD[d] for d in match.group(2))
    )


def fold_latin_to_ascii(text: str) -> str:
    text = str(text).lower().replace("\u2019", "").replace("\u2018", "").replace("`", "")
    for src, dst in UNICODE_EXPANSIONS.items():
        text = text.replace(src, dst)
    if not text.isascii():
        text = unicodedata.normalize("NFKD", text)
        text = text.encode("ascii", "ignore").decode("ascii")
    return text


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = fold_latin_to_ascii(text).replace("\r\n", "\n").replace("\r", "\n")
    if any(ch.isdigit() for ch in text):
        text = DECIMAL_RE.sub(_decimal_replacement, text)
        text = INTEGER_RE.sub(lambda m: integer_to_words(m.group(0)), text)
    text = text.translate(PUNCT_TRANSLATION)
    text = NON_MODEL_CHAR_RE.sub(" ", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n", text)
    return text.strip(" \n")


class CompactEnglishTokenizer:
    def __init__(self, core: Tokenizer) -> None:
        self.core = core
        self.vocab_size = core.get_vocab_size()
        self.special_ids: Dict[str, int] = {}
        for token in SPECIAL_TOKENS:
            tid = core.token_to_id(token)
            if tid is None:
                raise ValueError(f"tokenizer is missing control token {token}")
            self.special_ids[token] = int(tid)
        self.pad_id = self.special_ids["<pad>"]
        self.bos_id = self.special_ids["<bos>"]
        self.eos_id = self.special_ids["<eos>"]
        self.join_id = self.special_ids["<join>"]
        self.sent_id = self.special_ids["<sent>"]
        self.unk_id = self.special_ids["<unk>"]
        self.validate()

    @classmethod
    def load(cls, path: Path) -> "CompactEnglishTokenizer":
        return cls(Tokenizer.from_file(str(path)))

    def validate(self) -> None:
        if self.vocab_size != VOCAB_SIZE:
            raise ValueError(f"expected 5,000 tokenizer IDs, got {self.vocab_size}")
        specials = set(SPECIAL_TOKENS)
        bad = [
            token for token in self.core.get_vocab()
            if token not in specials and not LEXICAL_RE.fullmatch(token)
        ]
        if bad:
            raise ValueError(f"tokenizer contains nonalphabetic lexical pieces: {bad[:20]}")
        missing = [
            c for c in string.ascii_lowercase if self.core.token_to_id(c) is None
        ]
        if missing:
            raise ValueError("tokenizer is missing fallback letters: " + "".join(missing))

    def fingerprint(self) -> str:
        ordered = [self.core.id_to_token(i) or "" for i in range(self.vocab_size)]
        payload = json.dumps(
            {
                "normalization": NORMALIZATION_VERSION,
                "encoding": "implicit-space-join",
                "vocab": ordered,
            },
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def encode_normalized_batch(
        self,
        texts: Sequence[str],
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[List[int]]:
        encodings = self.core.encode_batch(list(texts), add_special_tokens=False)
        result: List[List[int]] = []
        for text, enc in zip(texts, encodings):
            ids: List[int] = [self.bos_id] if add_bos else []
            prev_end: Optional[int] = None
            for raw_id, (start, end) in zip(enc.ids, enc.offsets):
                tid = int(raw_id)
                if tid == self.unk_id:
                    raise RuntimeError("<unk> appeared after a-z normalization")
                if prev_end is not None:
                    if start == prev_end:
                        ids.append(self.join_id)
                    elif start > prev_end:
                        gap = text[prev_end:start]
                        if "\n" in gap:
                            ids.append(self.sent_id)
                ids.append(tid)
                prev_end = int(end)
            if add_eos:
                ids.append(self.eos_id)
            result.append(ids)
        return result

    def encode(self, text: str, *, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        normalized = normalize_text(text)
        return self.encode_normalized_batch(
            [normalized], add_bos=add_bos, add_eos=add_eos
        )[0]

    def decode(self, ids: Sequence[int], *, stop_at_eos: bool = True) -> str:
        out = ""
        join_next = False
        for raw in ids:
            tid = int(raw)
            if tid == self.eos_id:
                if stop_at_eos:
                    break
                join_next = False
                continue
            if tid in (self.pad_id, self.bos_id, self.unk_id):
                continue
            if tid == self.join_id:
                join_next = True
                continue
            if tid == self.sent_id:
                out = out.rstrip() + "\n"
                join_next = False
                continue
            token = self.core.id_to_token(tid)
            if token is None or not LEXICAL_RE.fullmatch(token):
                continue
            if not out or out.endswith("\n") or join_next:
                out += token
            else:
                out += " " + token
            join_next = False
        out = re.sub(r" +", " ", out)
        out = re.sub(r" *\n+ *", "\n", out)
        return out.strip()


# =============================================================================
# Model loading and inference
# =============================================================================


def trusted_torch_load(path: Path, *, map_location="cpu"):
    kwargs = {"map_location": map_location}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
    except Exception:
        pass
    return torch.load(path, **kwargs)


def amp_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def choose_amp(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def extract_npc_name(messages: Sequence[Mapping[str, str]]) -> Optional[str]:
    for msg in messages:
        if str(msg.get("role", "")).lower() not in ("system", "developer"):
            continue
        match = NAME_RE.search(str(msg.get("content", "")))
        if not match:
            continue
        normalized = normalize_text(match.group(1)).replace("\n", " ").strip()
        words = [w for w in normalized.split() if w]
        if words:
            return " ".join(words[:2])
    return None


def npc_speaker_cue(messages: Sequence[Mapping[str, str]]) -> str:
    name = extract_npc_name(messages)
    return f"{name} says" if name else PROMPT_CUES["assistant"]


def encode_prompt_for_next_npc(
    messages_before_response: Sequence[Mapping[str, str]],
    tok: CompactEnglishTokenizer,
) -> List[int]:
    user_cue_ids = tok.encode(PROMPT_CUES["user"])
    assistant_cue = npc_speaker_cue(messages_before_response)
    assistant_cue_ids = tok.encode(assistant_cue)
    ids: List[int] = [tok.bos_id]

    for msg in messages_before_response:
        role = ROLE_CANONICAL.get(str(msg["role"]).lower(), str(msg["role"]).lower())
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unsupported prompt role: {role}")
        if role == "system":
            ids.extend(tok.encode(str(msg["content"])))
            ids.append(tok.sent_id)
            continue
        cue_ids = user_cue_ids if role == "user" else assistant_cue_ids
        ids.extend(cue_ids)
        ids.append(tok.sent_id)
        ids.extend(tok.encode(str(msg["content"])))
        ids.append(tok.sent_id)

    ids.extend(assistant_cue_ids)
    ids.append(tok.sent_id)
    return ids


def sample_token(
    logits: torch.Tensor,
    *,
    forbidden: Sequence[int],
    temperature: float,
    top_k: int,
) -> torch.Tensor:
    scores = logits.float().clone()
    if forbidden:
        scores[:, list(forbidden)] = -float("inf")
    if temperature <= 0:
        return scores.argmax(dim=-1)
    scores /= max(temperature, 1e-6)
    if top_k > 0 and top_k < scores.shape[-1]:
        values, _ = torch.topk(scores, k=top_k, dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        scores = scores.masked_fill(scores < threshold, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate_npc_response(
    model: UOVPSMModel,
    tok: CompactEnglishTokenizer,
    messages_before_response: Sequence[Mapping[str, str]],
    *,
    device: torch.device,
    amp_dtype: Optional[torch.dtype],
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> Tuple[str, int]:
    prompt = encode_prompt_for_next_npc(messages_before_response, tok)
    x = torch.tensor([prompt], device=device, dtype=torch.long)
    reset = x.eq(tok.bos_id)

    model.eval()
    with amp_context(device, amp_dtype):
        out = model(x, state=None, reset_mask=reset, return_state=True)
    state = model.detach_state(out["state"])
    logits = out["logits"][:, -1, :]

    generated: List[int] = []
    forbidden = (tok.pad_id, tok.bos_id, tok.unk_id)
    for _ in range(max_new_tokens):
        nxt = sample_token(
            logits,
            forbidden=forbidden,
            temperature=temperature,
            top_k=top_k,
        )
        tid = int(nxt.item())
        generated.append(tid)
        if tid == tok.eos_id:
            break
        step_x = nxt.view(1, 1)
        with amp_context(device, amp_dtype):
            out = model(step_x, state=state, return_state=True)
        state = model.detach_state(out["state"])
        logits = out["logits"][:, -1, :]

    return tok.decode(generated, stop_at_eos=True), len(generated)


@dataclass
class LoadedModel:
    model: UOVPSMModel
    tokenizer: CompactEnglishTokenizer
    device: torch.device
    amp_dtype: Optional[torch.dtype]
    checkpoint_path: Path
    tokenizer_path: Path
    tokenizer_fingerprint: str
    parameter_count: int


def load_model(checkpoint_path: Path, tokenizer_path: Path) -> LoadedModel:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = choose_amp(device)

    tok = CompactEnglishTokenizer.load(tokenizer_path)
    tok_fp = tok.fingerprint()

    payload = trusted_torch_load(checkpoint_path, map_location="cpu")
    checkpoint_fp = payload.get("tokenizer_fingerprint")
    if checkpoint_fp not in (None, tok_fp):
        raise RuntimeError(
            "checkpoint tokenizer fingerprint does not match the supplied tokenizer"
        )

    config_dict = dict(payload.get("config", {}))
    if not config_dict:
        raise RuntimeError("checkpoint does not contain model config")
    config = UOVPSMConfig(**config_dict)
    if int(config.vocab_size) != VOCAB_SIZE:
        raise RuntimeError(
            f"checkpoint vocab is {config.vocab_size}, expected {VOCAB_SIZE}"
        )

    model = UOVPSMModel(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if hasattr(model, "num_parameters"):
        param_count = int(model.num_parameters())
    else:
        param_count = sum(p.numel() for p in model.parameters())

    return LoadedModel(
        model=model,
        tokenizer=tok,
        device=device,
        amp_dtype=amp_dtype,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        tokenizer_fingerprint=tok_fp,
        parameter_count=param_count,
    )


# =============================================================================
# Persona and identity protection
# =============================================================================


@dataclass(frozen=True)
class Persona:
    name: str
    profession: str
    town: str
    trait: str

    def normalized(self) -> "Persona":
        name = " ".join(self.name.strip().split()) or DEFAULT_NAME
        profession = " ".join(self.profession.strip().split()).lower() or DEFAULT_PROFESSION
        town = " ".join(self.town.strip().split()) or DEFAULT_TOWN
        trait = " ".join(self.trait.strip().split())
        if trait and trait[-1] not in ".!?":
            trait += "."
        if not trait:
            trait = DEFAULT_TRAIT
        return Persona(name=name, profession=profession, town=town, trait=trait)

    def system_prompt(self) -> str:
        p = self.normalized()
        article = indefinite_article(p.profession)
        return f"Your name is {p.name}. You are {article} {p.profession} from {p.town}. {p.trait}"


@dataclass
class GuardResult:
    text: str
    changed: bool
    reason: str = ""
    raw_text: str = ""
    intent: str = ""


def indefinite_article(noun: str) -> str:
    noun = noun.strip().lower()
    return "an" if noun[:1] in "aeiou" else "a"


def canonical_identity_reply(persona: Persona, intent: str) -> str:
    p = persona.normalized()
    name = p.name.lower()
    profession = p.profession.lower()
    town = p.town.lower()
    article = indefinite_article(profession)
    if intent == "name":
        return f"my name is {name}"
    if intent == "town":
        return f"i am from {town}"
    if intent == "trade":
        return f"i am {article} {profession}"
    return f"i am {name}, {article} {profession} from {town}"


def classify_identity_query(player_text: str) -> str:
    t = normalize_text(player_text).replace("\n", " ")

    full_patterns = (
        r"\bdescribe (?:yourself|thyself)\b",
        r"\btell me about (?:yourself|thyself)\b",
        r"\bwho art thou\b",
        r"\bwho are you\b",
        r"\bwhat are you\b",
    )
    name_patterns = (
        r"\bwhat is your name\b",
        r"\bwhats your name\b",
        r"\bwhat are you called\b",
        r"\bwho are you called\b",
        r"\btell me your name\b",
        r"\bwhat should i call you\b",
        r"\bwhat shall i call you\b",
        r"\bdo you remember your name\b",
    )
    town_patterns = (
        r"\bwhere are you from\b",
        r"\bwhat town are you from\b",
        r"\bwhich town are you from\b",
        r"\bwhere is your home\b",
        r"\bwhere do you live\b",
        r"\bwhich town do you call home\b",
        r"\bwhere were you raised\b",
    )
    trade_patterns = (
        r"\bwhat is your trade\b",
        r"\bwhats your trade\b",
        r"\bwhat is your profession\b",
        r"\bwhat do you do for work\b",
        r"\bwhat work do you do\b",
        r"\bhow do you earn your keep\b",
        r"\btell me your trade\b",
        r"\bwhat do you do\b",
    )

    for pattern in full_patterns:
        if re.search(pattern, t):
            return "full"
    for pattern in name_patterns:
        if re.search(pattern, t):
            return "name"
    for pattern in town_patterns:
        if re.search(pattern, t):
            return "town"
    for pattern in trade_patterns:
        if re.search(pattern, t):
            return "trade"

    # Identity corrections often omit the full question form.
    if any(word in t for word in ("your name", "thy name")):
        return "name"
    if "you are from" in t or "thy home" in t:
        return "town"
    if "your trade" in t or "your profession" in t:
        return "trade"
    return ""


def identity_answer_is_valid(text: str, persona: Persona, intent: str) -> bool:
    t = normalize_text(text).replace("\n", " ")
    p = persona.normalized()
    name = normalize_text(p.name).replace("\n", " ")
    town = normalize_text(p.town).replace("\n", " ")
    profession = normalize_text(p.profession).replace("\n", " ")

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


def _replace_explicit_self_claims(text: str, persona: Persona) -> Tuple[str, List[str]]:
    """Repair only explicit identity claims; ordinary prose is left untouched."""
    if not text.strip():
        return text, []

    p = persona.normalized()
    name = normalize_text(p.name).replace("\n", " ")
    town = normalize_text(p.town).replace("\n", " ")
    profession = normalize_text(p.profession).replace("\n", " ")
    article = indefinite_article(profession)
    out = text
    reasons: List[str] = []

    # Any phrase explicitly declaring a name is safe to canonicalize.
    patterns = [
        (r"\bmy name is\s+[a-z]+(?:\s+[a-z]+)?", f"my name is {name}", "name claim"),
        (r"\bi am called\s+[a-z]+(?:\s+[a-z]+)?", f"i am called {name}", "name claim"),
        (r"\bthey call me\s+[a-z]+(?:\s+[a-z]+)?", f"they call me {name}", "name claim"),
    ]
    for pattern, replacement, label in patterns:
        new = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
        if new != out:
            out = new
            reasons.append(label)

    # Correct "I am <known-name>" without touching "I am well/glad/tired".
    wrong_names = sorted(KNOWN_NAMES - {name}, key=len, reverse=True)
    if wrong_names:
        name_alt = "|".join(re.escape(v) for v in wrong_names)
        new = re.sub(
            rf"\bi am\s+(?:{name_alt})\b",
            f"i am {name}",
            out,
            flags=re.IGNORECASE,
        )
        if new != out:
            out = new
            reasons.append("known wrong name")

    # Correct explicit self-origin claims. This catches both "I am from Yew" and
    # "I am a merchant from Yew" while avoiding third-party place references.
    wrong_towns = sorted(KNOWN_TOWNS - {town}, key=len, reverse=True)
    if wrong_towns:
        town_alt = "|".join(re.escape(v) for v in wrong_towns)
        pattern = rf"(\bi am\b[^\n.!?]{{0,48}}\bfrom\s+)(?:{town_alt})\b"
        new = re.sub(pattern, lambda m: m.group(1) + town, out, flags=re.IGNORECASE)
        if new != out:
            out = new
            reasons.append("town claim")

    # Correct explicit self-profession claims. Custom professions are still
    # protected on direct identity queries; this list handles spontaneous drift.
    wrong_professions = sorted(KNOWN_PROFESSIONS - {profession}, key=len, reverse=True)
    if wrong_professions:
        prof_alt = "|".join(re.escape(v) for v in wrong_professions)
        pattern = rf"\bi am\s+(?:a|an)\s+(?:{prof_alt})\b"
        new = re.sub(pattern, f"i am {article} {profession}", out, flags=re.IGNORECASE)
        if new != out:
            out = new
            reasons.append("profession claim")

    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out).strip()
    return out, reasons


def protect_identity(
    player_text: str,
    raw_model_text: str,
    persona: Persona,
    *,
    enabled: bool = True,
) -> GuardResult:
    raw = (raw_model_text or "").strip()
    if not enabled:
        return GuardResult(text=raw, changed=False, raw_text=raw)

    intent = classify_identity_query(player_text)
    repaired, reasons = _replace_explicit_self_claims(raw, persona)

    if intent and not identity_answer_is_valid(repaired, persona, intent):
        canonical = canonical_identity_reply(persona, intent)
        return GuardResult(
            text=canonical,
            changed=(canonical != raw),
            reason=f"direct {intent} identity answer replaced",
            raw_text=raw,
            intent=intent,
        )

    if reasons:
        return GuardResult(
            text=repaired,
            changed=(repaired != raw),
            reason="repaired " + ", ".join(dict.fromkeys(reasons)),
            raw_text=raw,
            intent=intent,
        )

    return GuardResult(text=raw, changed=False, raw_text=raw, intent=intent)


# =============================================================================
# Chat GUI
# =============================================================================


class NPCChatGUI:
    def __init__(self, root: tk.Tk, args: argparse.Namespace) -> None:
        self.root = root
        self.args = args
        self.loaded: Optional[LoadedModel] = None
        self.guard_events: List[Dict[str, object]] = []
        self.turn_records: List[Dict[str, object]] = []
        self.busy = False
        self.load_queue: "queue.Queue[Tuple[str, object]]" = queue.Queue()

        self.root.title("VPSM v5 - Ultima Online NPC Chat")
        self.root.geometry("1160x820")
        self.root.minsize(940, 680)

        self._configure_style()
        self._build_ui()
        self._refresh_system_preview()
        self.root.after(80, self._poll_queue)
        self._start_model_load()

    # ------------------------------------------------------------------ UI --

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Header.TLabel", font=("Segoe UI", 13, "bold"))
        style.configure("Small.TLabel", font=("Segoe UI", 9))
        style.configure("TButton", padding=(8, 5))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(10, 6))

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="VPSM v5 NPC Chat", style="Header.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Loading model...")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        paned = ttk.Panedwindow(outer, orient="horizontal")
        paned.pack(fill="both", expand=True)

        chat_frame = ttk.Frame(paned, padding=(0, 0, 8, 0))
        side_frame = ttk.Frame(paned, padding=(8, 0, 0, 0))
        paned.add(chat_frame, weight=4)
        paned.add(side_frame, weight=2)

        # Chat transcript.
        self.chat = ScrolledText(
            chat_frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 11),
            padx=10,
            pady=10,
        )
        self.chat.pack(fill="both", expand=True)
        self.chat.tag_configure("player_label", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("npc_label", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("system", font=("Segoe UI", 9, "italic"))
        self.chat.tag_configure("body", spacing3=8)

        input_frame = ttk.Frame(chat_frame)
        input_frame.pack(fill="x", pady=(8, 0))
        self.input_box = tk.Text(input_frame, height=3, wrap="word", font=("Segoe UI", 11))
        self.input_box.pack(side="left", fill="x", expand=True)
        self.input_box.bind("<Return>", self._on_return)
        self.input_box.bind("<Shift-Return>", self._on_shift_return)
        self.send_button = ttk.Button(
            input_frame, text="Send", style="Accent.TButton", command=self.send_message
        )
        self.send_button.pack(side="left", padx=(8, 0), fill="y")

        # Persona panel.
        persona_box = ttk.LabelFrame(side_frame, text="NPC persona", padding=8)
        persona_box.pack(fill="x")

        self.name_var = tk.StringVar(value=self.args.name)
        self.profession_var = tk.StringVar(value=self.args.profession)
        self.town_var = tk.StringVar(value=self.args.town)
        self.trait_var = tk.StringVar(value=self.args.trait)

        self._labeled_entry(persona_box, "Name", self.name_var, 0)
        self._labeled_entry(persona_box, "Profession", self.profession_var, 1)
        self._labeled_entry(persona_box, "Town", self.town_var, 2)
        self._labeled_entry(persona_box, "Trait", self.trait_var, 3)
        for var in (self.name_var, self.profession_var, self.town_var, self.trait_var):
            var.trace_add("write", lambda *_: self._refresh_system_preview())

        buttons = ttk.Frame(persona_box)
        buttons.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(buttons, text="Apply + reset", command=self.apply_persona_reset).pack(side="left")
        ttk.Button(buttons, text="Random villager", command=self.randomize_persona).pack(side="left", padx=(6, 0))

        # Identity protection.
        guard_box = ttk.LabelFrame(side_frame, text="Identity protection", padding=8)
        guard_box.pack(fill="x", pady=(8, 0))
        self.guard_enabled_var = tk.BooleanVar(value=not self.args.no_identity_guard)
        ttk.Checkbutton(
            guard_box,
            text="Protect name / town / profession",
            variable=self.guard_enabled_var,
        ).pack(anchor="w")
        ttk.Label(
            guard_box,
            text=(
                "Direct identity mistakes are replaced. Explicit self-claims that drift "
                "to another town/trade/name are repaired in the displayed response."
            ),
            wraplength=310,
            style="Small.TLabel",
        ).pack(anchor="w", pady=(4, 4))
        self.guard_count_var = tk.StringVar(value="Corrections: 0")
        ttk.Label(guard_box, textvariable=self.guard_count_var).pack(anchor="w")

        # Generation settings.
        gen_box = ttk.LabelFrame(side_frame, text="Generation", padding=8)
        gen_box.pack(fill="x", pady=(8, 0))
        self.max_tokens_var = tk.IntVar(value=self.args.max_new_tokens)
        self.temperature_var = tk.DoubleVar(value=self.args.temperature)
        self.top_k_var = tk.IntVar(value=self.args.top_k)
        self._labeled_spin(gen_box, "Max tokens", self.max_tokens_var, 0, 4, 80, 1)
        self._labeled_spin(gen_box, "Temperature", self.temperature_var, 1, 0.0, 2.0, 0.05)
        self._labeled_spin(gen_box, "Top-k", self.top_k_var, 2, 0, 5000, 1)
        ttk.Label(
            gen_box,
            text=(
                "Temperature 0 = greedy. SINGLE-TURN MODE: every Send feeds only "
                "the persona system prompt and the current player message. Prior GUI "
                "turns are display-only and never enter model context."
            ),
            wraplength=310,
            style="Small.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(4, 0))

        # System prompt preview.
        sys_box = ttk.LabelFrame(side_frame, text="System prompt", padding=8)
        sys_box.pack(fill="x", pady=(8, 0))
        self.system_preview = tk.Text(sys_box, height=4, wrap="word", font=("Consolas", 9))
        self.system_preview.pack(fill="x")
        self.system_preview.configure(state="disabled")

        # Guard event log.
        log_box = ttk.LabelFrame(side_frame, text="Identity guard log", padding=8)
        log_box.pack(fill="both", expand=True, pady=(8, 0))
        self.guard_log = ScrolledText(log_box, height=8, wrap="word", font=("Consolas", 9), state="disabled")
        self.guard_log.pack(fill="both", expand=True)

        # Bottom actions.
        actions = ttk.Frame(side_frame)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Reset chat", command=self.reset_chat).pack(side="left")
        ttk.Button(actions, text="Save transcript", command=self.save_transcript).pack(side="left", padx=(6, 0))
        ttk.Button(actions, text="Reload model", command=self._start_model_load).pack(side="right")

        self.input_box.focus_set()

    def _labeled_entry(self, parent, label: str, variable: tk.StringVar, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", pady=2)
        parent.columnconfigure(1, weight=1)

    def _labeled_spin(self, parent, label, variable, row, low, high, step) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=2)
        spin = ttk.Spinbox(parent, textvariable=variable, from_=low, to=high, increment=step, width=10)
        spin.grid(row=row, column=1, sticky="e", pady=2)

    # --------------------------------------------------------------- Persona --

    def current_persona(self) -> Persona:
        return Persona(
            name=self.name_var.get(),
            profession=self.profession_var.get(),
            town=self.town_var.get(),
            trait=self.trait_var.get(),
        ).normalized()

    def _refresh_system_preview(self) -> None:
        if not hasattr(self, "system_preview"):
            return
        text = self.current_persona().system_prompt()
        self.system_preview.configure(state="normal")
        self.system_preview.delete("1.0", "end")
        self.system_preview.insert("1.0", text)
        self.system_preview.configure(state="disabled")

    def apply_persona_reset(self) -> None:
        if self.busy:
            return
        self.reset_chat()
        p = self.current_persona()
        self._append_system(f"Persona set: {p.name}, {p.profession}, {p.town}")

    def randomize_persona(self) -> None:
        if self.busy:
            return
        names = sorted(KNOWN_NAMES)
        professions = sorted(KNOWN_PROFESSIONS)
        towns = sorted(KNOWN_TOWNS)
        traits = [
            "You are practical and plain-spoken.",
            "You are friendly but not overly talkative.",
            "You are patient with strangers.",
            "You are a little gruff but fair.",
            "You are fond of company.",
            "You are steady and reliable.",
            "You are curious about the world.",
            "You are thoughtful and reserved.",
            "You are cheerful when work is light.",
            "You are cautious around trouble.",
            "You are kind to all you meet.",
            "You are proud of your work.",
        ]
        self.name_var.set(random.choice(names).title())
        self.profession_var.set(random.choice(professions))
        self.town_var.set(random.choice(towns).title())
        self.trait_var.set(random.choice(traits))
        self.reset_chat()

    # -------------------------------------------------------------- Loading --

    def _start_model_load(self) -> None:
        if self.busy:
            return
        self.busy = True
        self._set_send_enabled(False)
        self.status_var.set("Loading model...")
        checkpoint = Path(self.args.checkpoint)
        tokenizer = Path(self.args.tokenizer)

        def worker() -> None:
            try:
                loaded = load_model(checkpoint, tokenizer)
                self.load_queue.put(("loaded", loaded))
            except Exception:
                self.load_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self.load_queue.get_nowait()
                if kind == "loaded":
                    self.loaded = payload  # type: ignore[assignment]
                    self.busy = False
                    self._set_send_enabled(True)
                    amp = (
                        str(self.loaded.amp_dtype).replace("torch.", "")
                        if self.loaded and self.loaded.amp_dtype is not None
                        else "off"
                    )
                    dev = str(self.loaded.device) if self.loaded else "?"
                    self.status_var.set(
                        f"Ready | {dev} | {amp} | {self.loaded.parameter_count:,} params"
                    )
                    self._append_system(
                        f"Loaded {self.loaded.checkpoint_path.name} on {dev}. Identity guard is "
                        f"{'ON' if self.guard_enabled_var.get() else 'OFF'}."
                    )
                elif kind == "response":
                    self._finish_response(payload)  # type: ignore[arg-type]
                elif kind == "error":
                    self.busy = False
                    self._set_send_enabled(self.loaded is not None)
                    self.status_var.set("Error")
                    messagebox.showerror("VPSM error", str(payload))
        except queue.Empty:
            pass
        self.root.after(80, self._poll_queue)

    # --------------------------------------------------------------- Chat --

    def _on_return(self, event):
        if event.state & 0x0001:  # Shift is handled separately on Windows/Tk.
            return None
        self.send_message()
        return "break"

    def _on_shift_return(self, event):
        self.input_box.insert("insert", "\n")
        return "break"

    def send_message(self) -> None:
        if self.busy or self.loaded is None:
            return
        text = self.input_box.get("1.0", "end").strip()
        if not text:
            return
        self.input_box.delete("1.0", "end")

        persona = self.current_persona()
        self._append_chat("You", text, "player_label")
        model_messages = self._messages_for_model(persona, text)
        max_tokens = max(1, int(self.max_tokens_var.get()))
        temperature = max(0.0, float(self.temperature_var.get()))
        top_k = max(0, int(self.top_k_var.get()))
        guard_enabled = bool(self.guard_enabled_var.get())

        self.busy = True
        self._set_send_enabled(False)
        self.status_var.set("Thinking...")
        started = time.perf_counter()

        def worker() -> None:
            try:
                assert self.loaded is not None
                raw, token_count = generate_npc_response(
                    self.loaded.model,
                    self.loaded.tokenizer,
                    model_messages,
                    device=self.loaded.device,
                    amp_dtype=self.loaded.amp_dtype,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_k=top_k,
                )
                guard = protect_identity(
                    text,
                    raw,
                    persona,
                    enabled=guard_enabled,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.load_queue.put((
                    "response",
                    {
                        "player": text,
                        "raw": raw,
                        "guard": guard,
                        "persona": persona,
                        "elapsed_ms": elapsed_ms,
                        "tokens": token_count,
                    },
                ))
            except Exception:
                self.load_queue.put(("error", traceback.format_exc()))

        threading.Thread(target=worker, daemon=True).start()

    def _messages_for_model(
        self,
        persona: Persona,
        current_user: str,
    ) -> List[Dict[str, str]]:
        # Deliberately stateless across player turns. The production SFT target is
        # strongest when each query is evaluated as one independent NPC exchange.
        # The GUI keeps a visible transcript for the human, but none of that prior
        # transcript is included in the model prompt.
        return [
            {"role": "system", "content": persona.system_prompt()},
            {"role": "user", "content": current_user},
        ]

    def _finish_response(self, payload: Dict[str, object]) -> None:
        guard: GuardResult = payload["guard"]  # type: ignore[assignment]
        persona: Persona = payload["persona"]  # type: ignore[assignment]
        player = str(payload["player"])
        raw = str(payload["raw"])
        elapsed_ms = float(payload["elapsed_ms"])
        token_count = int(payload["tokens"])

        final_text = guard.text.strip()
        if not final_text:
            final_text = "..."

        self._append_chat(persona.name, final_text, "npc_label")

        record = {
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "persona": asdict(persona),
            "player": player,
            "raw_model": raw,
            "displayed": final_text,
            "identity_guard_changed": bool(guard.changed),
            "identity_guard_reason": guard.reason,
            "identity_intent": guard.intent,
            "generation_tokens": token_count,
            "latency_ms": round(elapsed_ms, 2),
        }
        self.turn_records.append(record)

        if guard.changed:
            event = dict(record)
            self.guard_events.append(event)
            self.guard_count_var.set(f"Corrections: {len(self.guard_events)}")
            self._append_guard_event(event)

        self.busy = False
        self._set_send_enabled(True)
        self.status_var.set(f"Ready | last response {elapsed_ms:.0f} ms | {token_count} tokens")
        self.input_box.focus_set()

    def reset_chat(self) -> None:
        if self.busy:
            return
        self.guard_events.clear()
        self.turn_records.clear()
        self.guard_count_var.set("Corrections: 0")
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self.guard_log.configure(state="normal")
        self.guard_log.delete("1.0", "end")
        self.guard_log.configure(state="disabled")
        self._append_system("Conversation reset.")

    def _append_chat(self, speaker: str, text: str, label_tag: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{speaker}\n", label_tag)
        self.chat.insert("end", text.strip() + "\n\n", "body")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _append_system(self, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"[{text}]\n\n", "system")
        self.chat.see("end")
        self.chat.configure(state="disabled")

    def _append_guard_event(self, event: Mapping[str, object]) -> None:
        raw = str(event.get("raw_model", ""))
        final = str(event.get("displayed", ""))
        reason = str(event.get("identity_guard_reason", ""))
        player = str(event.get("player", ""))
        block = (
            f"PLAYER: {player}\n"
            f"RAW:    {raw}\n"
            f"FIXED:  {final}\n"
            f"WHY:    {reason}\n"
            + ("-" * 38)
            + "\n"
        )
        self.guard_log.configure(state="normal")
        self.guard_log.insert("end", block)
        self.guard_log.see("end")
        self.guard_log.configure(state="disabled")

    def _set_send_enabled(self, enabled: bool) -> None:
        self.send_button.configure(state="normal" if enabled else "disabled")
        self.input_box.configure(state="normal" if enabled else "disabled")

    # -------------------------------------------------------------- Saving --

    def save_transcript(self) -> None:
        if not self.turn_records:
            messagebox.showinfo("Save transcript", "There is no conversation to save yet.")
            return
        suggested = f"vpsm_chat_{time.strftime('%Y%m%d_%H%M%S')}.json"
        path = filedialog.asksaveasfilename(
            title="Save VPSM chat transcript",
            defaultextension=".json",
            initialfile=suggested,
            filetypes=[("JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        loaded = self.loaded
        payload = {
            "persona": asdict(self.current_persona()),
            "system_prompt": self.current_persona().system_prompt(),
            "checkpoint": str(loaded.checkpoint_path) if loaded else self.args.checkpoint,
            "tokenizer": str(loaded.tokenizer_path) if loaded else self.args.tokenizer,
            "tokenizer_fingerprint": loaded.tokenizer_fingerprint if loaded else None,
            "identity_guard_enabled": bool(self.guard_enabled_var.get()),
            "inference_mode": "single_turn_system_plus_current_user_only",
            "temperature": float(self.temperature_var.get()),
            "top_k": int(self.top_k_var.get()),
            "max_new_tokens": int(self.max_tokens_var.get()),
            "turns": self.turn_records,
        }
        Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.status_var.set(f"Saved {Path(path).name}")


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="GUI chat client for VPSM v5 UO NPC model")
    p.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--tokenizer", type=str, default=str(DEFAULT_TOKENIZER))
    p.add_argument("--name", type=str, default=DEFAULT_NAME)
    p.add_argument("--profession", type=str, default=DEFAULT_PROFESSION)
    p.add_argument("--town", type=str, default=DEFAULT_TOWN)
    p.add_argument("--trait", type=str, default=DEFAULT_TRAIT)
    p.add_argument("--max-new-tokens", type=int, default=18)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--no-identity-guard", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    root = tk.Tk()
    app = NPCChatGUI(root, args)
    root.mainloop()


if __name__ == "__main__":
    main()
