from __future__ import annotations

"""
Reusable inference core for VPSM v5 UO NPC Final / v7-e14.

Important production rule:
    Each player utterance is an independent single-turn query.
    Do not feed previous player/NPC turns back into the model unless a future model
    is explicitly trained and validated for that use.
"""

import contextlib
import hashlib
import inspect
import json
import re
import string
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from tokenizers import Tokenizer

from .uo_vpsm_model_ren_v5_scan import UOVPSMConfig, UOVPSMModel


BUNDLE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = BUNDLE_ROOT / "model" / "vpsm_v5_uo_npc_final.pt"
DEFAULT_TOKENIZER = BUNDLE_ROOT / "tokenizer" / "tokenizer.json"

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
    "æ": "ae", "œ": "oe", "ß": "ss", "ø": "o",
    "ł": "l", "ð": "d", "þ": "th",
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
    text = str(text).lower().replace("’", "").replace("‘", "").replace("`", "")
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
        self.vocab_size = int(core.get_vocab_size())
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
    def load(cls, path: Path = DEFAULT_TOKENIZER) -> "CompactEnglishTokenizer":
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
        missing = [c for c in string.ascii_lowercase if self.core.token_to_id(c) is None]
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


def trusted_torch_load(path: Path, *, map_location="cpu"):
    kwargs = {"map_location": map_location}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
    except Exception:
        pass
    return torch.load(path, **kwargs)


def choose_amp(device: torch.device) -> Optional[torch.dtype]:
    if device.type != "cuda":
        return None
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def amp_context(device: torch.device, dtype: Optional[torch.dtype]):
    if device.type == "cuda" and dtype is not None:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return contextlib.nullcontext()


def indefinite_article(noun: str) -> str:
    noun = noun.strip().lower()
    return "an" if noun[:1] in "aeiou" else "a"


@dataclass(frozen=True)
class Persona:
    name: str = "Duncan"
    profession: str = "merchant"
    town: str = "Jhelom"
    trait: str = "You are proud of your work."

    def normalized(self) -> "Persona":
        name = " ".join(self.name.strip().split()) or "Duncan"
        profession = " ".join(self.profession.strip().split()).lower() or "merchant"
        town = " ".join(self.town.strip().split()) or "Jhelom"
        trait = " ".join(self.trait.strip().split())
        if trait and trait[-1] not in ".!?":
            trait += "."
        if not trait:
            trait = "You are proud of your work."
        return Persona(name=name, profession=profession, town=town, trait=trait)

    def system_prompt(self) -> str:
        p = self.normalized()
        article = indefinite_article(p.profession)
        return f"Your name is {p.name}. You are {article} {p.profession} from {p.town}. {p.trait}"


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
    scores /= max(float(temperature), 1e-6)
    if top_k > 0 and top_k < scores.shape[-1]:
        values, _ = torch.topk(scores, k=int(top_k), dim=-1)
        threshold = values[:, -1].unsqueeze(-1)
        scores = scores.masked_fill(scores < threshold, -float("inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@dataclass
class FinalNPCModel:
    model: UOVPSMModel
    tokenizer: CompactEnglishTokenizer
    device: torch.device
    amp_dtype: Optional[torch.dtype]
    checkpoint_path: Path
    tokenizer_path: Path
    tokenizer_fingerprint: str
    parameter_count: int
    checkpoint_epoch: int


def load_final_model(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    tokenizer_path: Path = DEFAULT_TOKENIZER,
    *,
    device: Optional[str] = None,
) -> FinalNPCModel:
    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer not found: {tokenizer_path}")

    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    amp_dtype = choose_amp(dev)

    tok = CompactEnglishTokenizer.load(tokenizer_path)
    tok_fp = tok.fingerprint()

    payload = trusted_torch_load(checkpoint_path, map_location="cpu")
    checkpoint_fp = payload.get("tokenizer_fingerprint")
    if checkpoint_fp not in (None, tok_fp):
        raise RuntimeError("checkpoint tokenizer fingerprint does not match tokenizer")

    config_dict = dict(payload.get("config", {}))
    if not config_dict:
        raise RuntimeError("checkpoint does not contain model config")
    config = UOVPSMConfig(**config_dict)
    if int(config.vocab_size) != VOCAB_SIZE:
        raise RuntimeError(f"checkpoint vocab is {config.vocab_size}, expected {VOCAB_SIZE}")

    model = UOVPSMModel(config).to(dev)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if hasattr(model, "num_parameters"):
        param_count = int(model.num_parameters())
    else:
        param_count = int(sum(p.numel() for p in model.parameters()))

    return FinalNPCModel(
        model=model,
        tokenizer=tok,
        device=dev,
        amp_dtype=amp_dtype,
        checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
        tokenizer_fingerprint=tok_fp,
        parameter_count=param_count,
        checkpoint_epoch=int(payload.get("epoch", -1)),
    )


@torch.no_grad()
def generate_single_turn(
    loaded: FinalNPCModel,
    player_text: str,
    persona: Persona,
    *,
    max_new_tokens: int = 18,
    temperature: float = 0.0,
    top_k: int = 0,
) -> Tuple[str, int]:
    """Generate one response with a fresh recurrent state for this query only."""
    text = str(player_text).strip()
    if not text:
        return "", 0

    p = persona.normalized()
    messages = [
        {"role": "system", "content": p.system_prompt()},
        {"role": "user", "content": text},
    ]

    tok = loaded.tokenizer
    model = loaded.model
    device = loaded.device
    prompt = encode_prompt_for_next_npc(messages, tok)
    x = torch.tensor([prompt], device=device, dtype=torch.long)
    reset = x.eq(tok.bos_id)

    # Critical: state=None means each query starts fresh. We intentionally do not
    # carry state or old chat text from one player utterance to the next.
    with amp_context(device, loaded.amp_dtype):
        out = model(x, state=None, reset_mask=reset, return_state=True)
    state = model.detach_state(out["state"])
    logits = out["logits"][:, -1, :]

    generated: List[int] = []
    forbidden = (tok.pad_id, tok.bos_id, tok.unk_id)
    for _ in range(int(max_new_tokens)):
        nxt = sample_token(
            logits,
            forbidden=forbidden,
            temperature=float(temperature),
            top_k=int(top_k),
        )
        tid = int(nxt.item())
        generated.append(tid)
        if tid == tok.eos_id:
            break
        step_x = nxt.view(1, 1)
        with amp_context(device, loaded.amp_dtype):
            out = model(step_x, state=state, return_state=True)
        state = model.detach_state(out["state"])
        logits = out["logits"][:, -1, :]

    return tok.decode(generated, stop_at_eos=True), len(generated)
