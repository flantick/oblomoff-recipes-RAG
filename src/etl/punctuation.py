"""Restoring punctuation and casing in ASR transcripts (Step 1.5).

The task is solved as sequence labeling rather than generation: the number of
words is preserved, so every word stays accurately bound to the timecode of its
cue.

Backends:
    rupunct : token-classification BERT (RUPunct/RUPunct_big|small), GPU batches
    silero  : torch.hub silero_te, CPU, zero setup (fallback)

restore_cues() returns a list of Sentence(text, start, end), where start is the
timestamp of the cue in which the first word of the sentence begins.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from loguru import logger

from src.etl.cleaning import clean_line
from src.etl.schemas import RawCue

# --- decoder for the RUPunct labels ----------------------------------
_CASE_FN = {
    "LOWER": lambda t: t,
    "UPPER": lambda t: t[:1].upper() + t[1:],
    "UPPER_TOTAL": lambda t: t.upper(),
}
_PUNCT_SUFFIX = {
    "O": "",
    "PERIOD": ".",
    "COMMA": ",",
    "QUESTION": "?",
    "TIRE": " —",
    "DVOETOCHIE": ":",
    "VOSKL": "!",
    "PERIODCOMMA": ";",
    "DEFIS": "-",
    "MNOGOTOCHIE": "...",
    "QUESTIONVOSKL": "?!",
}
_SENT_FINAL = (".", "?", "!", "…")


def _decode_label(token: str, label: str) -> str:
    for case_key in ("UPPER_TOTAL", "UPPER", "LOWER"):
        if label.startswith(case_key + "_"):
            punct_key = label[len(case_key) + 1:]
            case_fn = _CASE_FN[case_key]
            return case_fn(token) + _PUNCT_SUFFIX.get(punct_key, "")
    return token


@dataclass
class Sentence:
    text: str
    start: float
    end: float


# --- assembling sentences out of labelled words -------------------
def _assemble_sentences(
    decoded: list[str],
    spans: list[tuple[float, float]],
    hard_max_chars: int,
) -> list[Sentence]:
    sentences: list[Sentence] = []
    buf: list[str] = []
    buf_start: float | None = None
    buf_end: float = 0.0

    def flush() -> None:
        nonlocal buf, buf_start, buf_end
        if not buf or buf_start is None:
            buf, buf_start = [], None
            return
        text = " ".join(buf)
        text = re.sub(r"\s+([,.!?…:;])", r"\1", text).strip()
        text = re.sub(r"\s{2,}", " ", text)
        if text:
            text = text[:1].upper() + text[1:]
            sentences.append(Sentence(text=text, start=round(buf_start, 2), end=round(buf_end, 2)))
        buf, buf_start = [], None

    for tok, (s, e) in zip(decoded, spans):
        if not tok:
            continue
        if buf_start is None:
            buf_start = s
        buf.append(tok)
        buf_end = e
        cur_len = sum(len(x) + 1 for x in buf)
        if tok.rstrip().endswith(_SENT_FINAL):
            flush()
        elif cur_len >= hard_max_chars and tok.rstrip().endswith(","):
            # the model missed a full stop — break on a comma to avoid a wall of text
            flush()

    flush()
    return sentences


# --- two-pass alignment (for silero) -----------------------------
_NORM_RE = re.compile(r"[^\w-]", re.UNICODE)


def _norm(s: str) -> str:
    return _NORM_RE.sub("", s).lower()


def _align_restored(original: list[str], restored_tokens: list[str]) -> list[str]:
    """Aligns the string restored by silero back onto the original word list.
    The order and the number of words practically match; on a desync we fall back
    to the original word."""
    out: list[str] = []
    j = 0
    n = len(restored_tokens)
    for w in original:
        target = _norm(w)
        if j < n and _norm(restored_tokens[j]) == target:
            out.append(restored_tokens[j])
            j += 1
        elif j + 1 < n and _norm(restored_tokens[j + 1]) == target:
            j += 2
            out.append(restored_tokens[j - 1])
        else:
            out.append(w)
            if j < n:
                j += 1
    return out


# --- the main class ------------------------------------------------
class PunctuationRestorer:
    def __init__(
        self,
        backend: str = "auto",
        *,
        model_name: str = "RUPunct/RUPunct_big",
        device: str | None = None,
        window: int = 150,
        stride: int = 110,
        hard_max_chars: int = 400,
    ) -> None:
        self.requested_backend = backend
        self.backend = backend
        self.model_name = model_name
        self.window = window
        self.stride = max(1, min(stride, window - 1))
        self.hard_max_chars = hard_max_chars
        self._tok = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._silero_apply = None
        self._torch = None

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        self.device = device

        if backend in ("auto", "rupunct"):
            try:
                self._init_rupunct()
                self.backend = "rupunct"
                return
            except Exception as exc:  # noqa: BLE001
                if backend == "rupunct":
                    raise
                logger.warning("RUPunct недоступен ({}), пробую silero", exc)
        if backend in ("auto", "silero"):
            try:
                self._init_silero()
                self.backend = "silero"
                return
            except Exception as exc:  # noqa: BLE001
                if backend == "silero":
                    raise
                logger.warning("silero недоступен ({}) — пунктуация отключена", exc)
        self.backend = "none"

    # --- backend initialisation --------------------------------
    def _init_rupunct(self) -> None:
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        if not self._tok.is_fast:
            raise RuntimeError("нужен fast-токенизатор (word_ids)")
        self._model = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._model.to(self.device).eval()
        self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}
        logger.info("RUPunct загружен: {} на {}", self.model_name, self.device)

    def _init_silero(self) -> None:
        import torch

        self._torch = torch
        model_data = torch.hub.load(
            repo_or_dir="snakers4/silero-models", model="silero_te", trust_repo=True
        )
        # (model, example_texts, languages, punct, apply_te)
        self._silero_apply = model_data[-1]
        logger.info("silero_te загружен (CPU)")

    # --- public API -------------------------------------------
    def restore_cues(self, cues: list[RawCue], *, pre_clean: bool = True) -> list[Sentence]:
        words: list[str] = []
        spans: list[tuple[float, float]] = []
        for c in cues:
            text = clean_line(c.text) if pre_clean else c.text
            if not text:
                continue
            end = c.start + c.duration
            for w in text.split():
                words.append(w)
                spans.append((c.start, end))
        if not words:
            return []

        if self.backend == "rupunct":
            decoded = self._label_rupunct(words)
        elif self.backend == "silero":
            decoded = self._label_silero(words)
        else:
            decoded = words  # no punctuation — at least split on the pauses

        return _assemble_sentences(decoded, spans, self.hard_max_chars)

    # --- RUPunct inference -----------------------------------
    def _label_rupunct(self, words: list[str]) -> list[str]:
        torch = self._torch
        n = len(words)
        # a label_id per word + how central the window it was taken from was
        best_label = [None] * n
        best_centrality = [-1.0] * n

        starts = list(range(0, n, self.stride)) or [0]
        windows = [(s, min(s + self.window, n)) for s in starts]

        for ws, we in windows:
            chunk = words[ws:we]
            enc = self._tok(
                chunk,
                is_split_into_words=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            word_ids = enc.word_ids(0)
            with torch.no_grad():
                logits = self._model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            pred = logits.argmax(-1)[0].tolist()

            # Only the first sub-token of a word can win: centrality depends on
            # wid alone and the comparison below is strict, so every later
            # sub-token of the same word ties and loses.
            for pos, wid in enumerate(word_ids):
                if wid is None:
                    continue
                gidx = ws + wid
                if gidx >= n:
                    continue
                centrality = 1.0 - abs((wid - (we - ws) / 2)) / max((we - ws) / 2, 1)
                if centrality > best_centrality[gidx]:
                    best_centrality[gidx] = centrality
                    best_label[gidx] = pred[pos]

        out: list[str] = []
        for w, lid in zip(words, best_label):
            label = self._id2label.get(int(lid), "LOWER_O") if lid is not None else "LOWER_O"
            out.append(_decode_label(w, label))
        return out

    # --- silero inference ----------------------------------
    def _label_silero(self, words: list[str]) -> list[str]:
        # windowed, so that silero does not truncate a long input
        restored_all: list[str] = []
        step = 400
        for i in range(0, len(words), step):
            piece = " ".join(words[i:i + step])
            try:
                res = self._silero_apply(piece, lan="ru")
            except Exception:  # noqa: BLE001
                res = piece
            restored_all.extend(res.split())
        return _align_restored(words, restored_all)
