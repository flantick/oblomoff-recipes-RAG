"""Single configuration entry point for the whole pipeline (Step 1)."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / os.getenv("DATA_RAW_DIR", "data/raw")
DATA_PROCESSED_DIR = ROOT_DIR / os.getenv("DATA_PROCESSED_DIR", "data/processed")
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Network settings ---------------------------------------------------
ETL_SLEEP_SECONDS = float(os.getenv("ETL_SLEEP_SECONDS", "1.0"))
SUB_LANGS = [s.strip() for s in os.getenv("ETL_SUB_LANGS", "ru,ru-RU").split(",") if s.strip()]
PROXY = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY") or None

# Working around YouTube rate limits: cookies from a logged-in browser plus
# pauses for yt-dlp. The --cookies-from-browser / --cookies-file CLI flags
# override these values.
YTDLP_COOKIES_FROM_BROWSER = os.getenv("YTDLP_COOKIES_FROM_BROWSER") or None  # chrome|firefox|edge|brave|...
YTDLP_COOKIES_FILE = os.getenv("YTDLP_COOKIES_FILE") or None                  # path to cookies.txt (Netscape)
# YouTube serves AUTO-TRANSLATED video titles (inside playlists especially), so
# half of the corpus used to arrive with titles like "Electric Rotisserie Grill"
# instead of the original Russian one. Ask for the original.
YTDLP_LANG = os.getenv("YTDLP_LANG", "ru")
YTDLP_SLEEP_SUBTITLES = float(os.getenv("YTDLP_SLEEP_SUBTITLES", "2"))
YTDLP_RETRIES_429 = int(os.getenv("YTDLP_RETRIES_429", "1"))  # long backoffs are useless against a hard IP block

# --- ASR: local Whisper instead of YouTube subtitles ---------------------
# timedtext (the subtitle endpoint) is throttled per IP, and neither cookies nor
# a PO token fix that; the media CDN meanwhile stays open. So we download the
# audio and transcribe it ourselves.
ASR_MODEL = os.getenv("ASR_MODEL", "large-v3-turbo")
ASR_DEVICE = os.getenv("ASR_DEVICE") or None            # cuda | cpu (auto by default)
ASR_COMPUTE_TYPE = os.getenv("ASR_COMPUTE_TYPE") or None  # float16 on GPU, int8 on CPU
ASR_BATCH_SIZE = int(os.getenv("ASR_BATCH_SIZE", "16"))
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "ru")
# Channel videos carry an auto-dubbed English track, and it comes FIRST in the
# format list: without an explicit language filter that is what gets downloaded
# instead of the original.
ASR_AUDIO_FORMAT = os.getenv(
    "ASR_AUDIO_FORMAT",
    "ba[language^=ru][protocol^=http]/ba[protocol^=http][language=none]/ba[protocol^=http]/ba",
)
ASR_KEEP_AUDIO = os.getenv("ASR_KEEP_AUDIO", "0") == "1"
_asr_dir = os.getenv("ASR_AUDIO_DIR")
ASR_AUDIO_DIR = (ROOT_DIR / _asr_dir) if _asr_dir else None  # None => temporary directory

# --- Playlists of the oblomoffood channel ------------------------------
PLAYLISTS: list[str] = [
    "https://youtube.com/playlist?list=PLJFORpGMfFRh2GXeygxm-eAuyf1MVxonD",
    "https://youtube.com/playlist?list=PLJFORpGMfFRiieGF84brRztPs-6IG2zVR",
    "https://youtube.com/playlist?list=PLJFORpGMfFRgKydPm5QEMwcaewIaZDFne",
    "https://youtube.com/playlist?list=PLJFORpGMfFRgTEj8P9HYyZewf8GROOSbx",
    "https://youtube.com/playlist?list=PLJFORpGMfFRgjYgTzvZKbm6u0kjm6eKK9",
    "https://youtube.com/playlist?list=PLJFORpGMfFRjPJfkZFn1WcpElYkhbDWfg",
    "https://youtube.com/playlist?list=PLJFORpGMfFRiVorbAG9J1zmYIGJkJ7_1d",
    "https://youtube.com/playlist?list=PLJFORpGMfFRijzxmF6MmGauKubdBAgSrl",
    "https://youtube.com/playlist?list=PLJFORpGMfFRhlnZxY1_0cjJ8WrRKH7pnW",
]

# --- Text cleaning ----------------------------------------------------
# Filler words (removed only as standalone tokens, on word boundaries).
# The list is deliberately conservative: meaningful connectives ("значит",
# "получается") are left alone so that recipe steps stay intact.
FILLER_WORDS: list[str] = [
    "ну", "вот", "как бы", "типа", "короче", "собственно",
    "так сказать", "это самое", "в общем-то", "как говорится",
    "блин", "чё", "чо",
]

# Interjections / pause fillers.
INTERJECTIONS: list[str] = [
    "э", "ээ", "эээ", "эм", "эмм", "ммм", "мм", "аа", "ааа",
    "ну-ну", "э-э", "э-э-э", "а-а", "а-а-а",
]

# Markers of sponsored integrations. On a match, a window around the hit is cut
# out (see cleaning.AD_WINDOW_*). Everything removed is logged.
# These are substrings (IGNORECASE, no word boundaries): stems are used so that
# every word form is caught, e.g. "ссылочк" -> ссылочки/ссылочка/ссылочку.
AD_MARKERS: list[str] = [
    "промокод", "erid", "рекламодател", "спонсор", "при поддержке",
    "партнёр выпуска", "партнер выпуска", "наш партнёр", "наш партнер",
    "ссылка в описании", "ссылочк", "ссылки в описании", "по ссылке ниже",
    "переходите по ссылке", "скидка по ссылке", "скидку по промокоду",
    "кэшбэк", "кешбэк", "промо-код", "закреплённом комментарии",
    "закрепленном комментарии", "устанавливайте приложение", "качайте приложение",
    "скачивайте приложение",
]

# Markers of non-recipe videos (food delivery reviews and the like) —
# a heuristic over title/description.
DELIVERY_MARKERS: list[str] = [
    "обзор доставки", "доставка еды", "заказал доставку", "заказали доставку",
    "проверка доставки", "доставка из", "доставки на дом",
]

# --- Parameters for merging cues into semantic blocks ---------------
BLOCK_TARGET_CHARS = 400      # size we aim for when flushing a block
BLOCK_MAX_CHARS = 750         # hard block size limit
BLOCK_GAP_SECONDS = 3.0       # speech pause that forces a block break

# --- Step 2: chunking ------------------------------------------------
CHUNK_TARGET_TOKENS = 450     # target chunk size
CHUNK_MAX_TOKENS = 600        # hard limit (may be exceeded by one sentence at most)
CHUNK_MIN_TOKENS = 120        # smaller chunks are glued to a neighbour
CHUNK_OVERLAP_TOKENS = 80     # overlap between adjacent chunks (except at section borders)
CHARS_PER_TOKEN = 2.5         # token-count heuristic for Cyrillic (when --tokenizer is absent)

# Units of measure / quantities — markers of the "ingredients" section.
INGREDIENT_UNIT_RE = (
    r"\d+(?:[.,]\d+)?\s*"
    r"(?:г|гр|грамм\w*|кг|мл|л|литр\w*|ст\.?\s?л\.?|стол\w*\s+ложк\w*|"
    r"ч\.?\s?л\.?|чайн\w*\s+ложк\w*|шт\.?|штук\w*|зубчик\w*|стакан\w*|"
    r"щепот\w*|горст\w*|долек|долько\w*|ломт\w*|пучок|пучк\w*|банк\w*|уп\.?|пачк\w*)"
)
INGREDIENT_PHRASES = ["по вкусу", "на кончике ножа", "щепотка", "по желанию"]

# Action verb stems — markers of the "steps" section.
STEP_VERBS = [
    "нареж", "режем", "режьте", "обжар", "жар", "туш", "варим", "варить", "отвар",
    "запек", "запеч", "смешай", "смешива", "перемеш", "добав", "посол", "поперч",
    "посыпь", "натри", "натёр", "натер", "выклад", "выложи", "полей", "залей",
    "взбей", "взбива", "маринуй", "марин", "разогрей", "разогрева", "поставь",
    "ставим", "накрой", "остуд", "нагрей", "нагрева",
]
SEQ_MARKERS = [
    "сначала", "затем", "потом", "далее", "после этого", "после чего", "теперь",
    "следующим шагом", "в конце", "в самом конце", "параллельно", "тем временем",
    "как только", "когда", "первым делом",
]
INTRO_MARKERS = [
    "всем привет", "здравствуйте", "дорогие друзья", "друзья", "сегодня приготов",
    "сегодня будем готов", "сегодня я приготов", "в этом видео", "в этом ролике",
    "меня зовут", "с вами", "сегодня у нас",
]
OUTRO_MARKERS = [
    "приятного аппетита", "ставьте лайк", "ставьте лайки", "подписывайтесь",
    "подпишись", "до новых встреч", "до новых видео", "всем пока", "до встречи",
    "приятного просмотра", "спасибо за просмотр", "пишите в комментар",
]

# --- Step 3: embeddings + vector database -----------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
EMBED_DEVICE = os.getenv("EMBED_DEVICE") or None       # None -> auto (cuda, else cpu)
EMBED_USE_FP16 = os.getenv("EMBED_USE_FP16", "1") == "1"
EMBED_MAX_LENGTH = int(os.getenv("EMBED_MAX_LENGTH", "1024"))
DENSE_VECTOR_SIZE = 1024                               # bge-m3

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "oblomoff_recipes")

# --- Step 4: retrieval -------------------------------------------
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_DEVICE = os.getenv("RERANK_DEVICE") or None      # "cpu" while the GPU is busy with vLLM
# A whole recipe lives in ONE video and is stretched over its entire runtime, so
# we take few videos but go deep into each. With top_videos=3/per_video=2 the
# context ended up holding two intro chunks (the dish name is said there most
# often) — and the LLM honestly answered found=false, having seen neither
# ingredients nor steps.
RETRIEVAL_TOP_VIDEOS = int(os.getenv("RETRIEVAL_TOP_VIDEOS", "2"))
RETRIEVAL_PER_VIDEO = int(os.getenv("RETRIEVAL_PER_VIDEO", "6"))
RETRIEVAL_OVERFETCH = int(os.getenv("RETRIEVAL_OVERFETCH", "6"))
RETRIEVAL_RERANK_TOP = int(os.getenv("RETRIEVAL_RERANK_TOP", "24"))  # lower => faster reranking on CPU
# max_model_len=8192 minus an answer of up to LLM_MAX_TOKENS(1200) leaves ~6k
# for the context.
RETRIEVAL_TOKEN_BUDGET = int(os.getenv("RETRIEVAL_TOKEN_BUDGET", "5200"))
# Safety valve against an oversized passage. 900 words ~= 1400 tokens, so a
# stitch of 2 adjacent chunks passes through whole (the median chunk is 274
# words). Larger values are unsafe: a single oversized block ate a quarter of the
# budget and left no room for the middle of the recipe.
RETRIEVAL_MAX_PASSAGE_WORDS = int(os.getenv("RETRIEVAL_MAX_PASSAGE_WORDS", "500"))
# Pull in the neighbours of the retrieved chunks: a recipe is continuous in time,
# while ranking picks out isolated points from it.
RETRIEVAL_NEIGHBOR_RADIUS = int(os.getenv("RETRIEVAL_NEIGHBOR_RADIUS", "1"))
# Per-section score multipliers. The signal is weak: on an ASR corpus the section
# labelling is nearly degenerate (steps 78%, ingredients 0.9%), and the first
# chunk of a video is tagged intro for only a fifth of the videos — the author
# starts using action verbs from the very first minute.
RETRIEVAL_SECTION_WEIGHTS: dict[str, float] = {
    "intro": 0.35,
    "outro": 0.25,
    "ingredients": 1.15,
}
# So the main penalty is positional, which is objective. The first chunk of a
# video is a greeting plus an enumeration (25 mentions of the dish name and not a
# single instruction): a perfect lexical match and pure noise in the context.
RETRIEVAL_FIRST_CHUNK_WEIGHT = float(os.getenv("RETRIEVAL_FIRST_CHUNK_WEIGHT", "0.30"))
RETRIEVAL_LAST_CHUNK_WEIGHT = float(os.getenv("RETRIEVAL_LAST_CHUNK_WEIGHT", "0.35"))

# Query keywords -> substring of a playlist title (soft intent filter).
PLAYLIST_INTENTS: list[tuple[str, str]] = [
    (r"сала[тд]", "Салаты"),
    (r"\bсуп\w*|борщ|щи\b|солянк", "Супы"),
    (r"ры[бо]\w*|лосос|форел|треск|креветк|кальмар|мидии|морепродукт", "Блюда из рыбы и морепродуктов"),
    (r"закуск|намазк|паштет|брускетт", "Закуски"),
    (r"кур\w*|куриц|индейк|утк\w*|птиц", "Блюдо из птицы"),
    (r"стейк|говядин|свинин|баранин|мяс\w*|фарш|котлет", "Блюда из мяса"),
    (r"шашлык|мангал|гриль|барбекю|бербекю|на угл\w*|на костре", "Мангал, гриль, бербекю, печь"),
    (r"бутер\w*|сэндвич|сендвич|тост\w*", "Бутеры"),
    (r"новогодн|на новый год|рождествен", "Новогодние рецепты"),
]

# --- Step 5: generation (vLLM, OpenAI-compatible API) -------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")          # vLLM does not check the key
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5-14b")        # = --served-model-name
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1200"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
