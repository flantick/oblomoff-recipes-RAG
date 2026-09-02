# oblomoff recipes RAG — рецепты с youtube «oblomoffood»

RAG-система: задаёшь вопрос («как приготовить сочный стейк?») — получаешь
структурированный рецепт (блюдо / ингредиенты / шаги) **только из того, что
реально сказано в видео**, со ссылкой на ролик и таймкод.

## Стек

| Слой | Технология |
|---|---|
| ETL | `yt-dlp` + локальный ASR (`faster-whisper`), чистка, вырезание рекламы |
| Чанкинг | семантические блоки по предложениям + разметка секций рецепта |
| Эмбеддинги | `BAAI/bge-m3` (dense 1024 + sparse/lexical из одной модели) |
| Векторная БД | Qdrant (named vector `dense` + sparse `lexical`) |
| Retrieval | гибридный поиск (RRF) → ре-ранк `bge-reranker-v2-m3` → дедуп по видео → сборка контекста |
| Генерация | vLLM + `Qwen2.5-14B-Instruct-AWQ` (OpenAI-совместимый API) |
| API / UI | FastAPI, опционально Streamlit |

## Быстрый старт (Docker)

Требуется NVIDIA GPU 16 ГБ + nvidia-container-toolkit.

```bash
cp .env.example .env

# 1. поднять БД и LLM
docker compose up -d qdrant vllm            # vLLM грузит ~10 ГБ весов, ~5 мин

# 2. индексация (один раз). Быстро — на GPU с хоста, погасив vLLM:
docker compose stop vllm
docker compose up -d pot-provider                    # PO-token для yt-dlp
python -m src.etl.pipeline --source asr --drop-non-recipe   # см. раздел ETL ниже
python -m src.chunking.run --tokenizer BAAI/bge-m3
EMBED_DEVICE=cuda python -m src.index.build --recreate
docker compose start vllm
#   …либо целиком в контейнере на CPU (медленно):
#   docker compose --profile tools run --rm indexer

# 3. API
docker compose up -d api                    # http://localhost:8080/docs

# 4. (опц.) веб-интерфейс
docker compose --profile ui up -d ui        # http://localhost:8501
```

Проверка:

```bash
curl localhost:8080/health
curl -X POST localhost:8080/ask -H 'content-type: application/json' \
  -d '{"query":"маринад для курицы на мангале"}'
```

## Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| `GET` | `/health` | статус Qdrant и vLLM |
| `GET` | `/search?q=&k=&mode=` | сырой retrieval (без LLM), быстрый |
| `POST` | `/ask` | `{query, top_videos?, per_video?, use_intent_filter?}` → `RecipeAnswer` |

## Локальный запуск без Docker

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
pip install "torch>=2.7" --index-url https://download.pytorch.org/whl/cu128   # своя CUDA

docker compose up -d qdrant vllm
EMBED_DEVICE=cpu RERANK_DEVICE=cpu uvicorn src.api.main:app --port 8080
```

## Пайплайн по шагам

```
src/etl/         Шаг 1   плейлисты → транскрипты с таймкодами → чистка
src/etl/asr.py          Шаг 1.5  аудиодорожка → faster-whisper (основной путь)
src/etl/punctuation.py  Шаг 1.5  RUPunct: пунктуация + регистр — только для пути с субтитрами
src/chunking/    Шаг 2   транскрипт → чанки (предложения, секции, метаданные)
src/index/       Шаг 3   BGE-M3 → Qdrant (dense + sparse), идемпотентный upsert
src/retrieval/   Шаг 4   гибрид + ре-ранк + группировка + контекст
src/generation/  Шаг 5   промпт → vLLM → RecipeAnswer (grounded, с цитатами)
src/api/         Шаг 6   FastAPI
src/ui/          Шаг 6   Streamlit (опц.)
```

Каждый модуль запускается как CLI, напр.:

```bash
python -m src.retrieval.retrieve "маринад для шашлыка"
python -m src.generation.ask "как замариновать курицу" --json
```

## ETL: почему ASR, а не субтитры

Изначально транскрипты брались готовыми из YouTube. Это перестало работать:
эндпоинт субтитров `timedtext` жёстко лимитируется **по IP**, и после нескольких
сотен запросов отдаёт `IpBlocked` / `HTTP 429` на каждый вызов. Блок держится
сутки и снимается всего на ~20 запросов. Не помогают ни куки залогиненного
аккаунта, ни смена `player_client`, ни паузы между запросами.

При этом блокировка бьёт только по субтитрам. Остальные слои живы:

| Слой | Состояние | Решение |
|---|---|---|
| `timedtext` (текст субтитров) | жёсткий per-IP 429 | обходим — не используем |
| InnerTube player API (плейлисты, метаданные) | challenge `The page needs to be reloaded` | PO-token (`pot-provider` в compose) |
| медиа-CDN `googlevideo` (аудио) | без ограничений | качаем аудио |

Поэтому основной путь — `--source asr`: качаем оригинальную аудиодорожку и
распознаём её локально на GPU (`large-v3-turbo`, ~90x realtime). Заодно вышло
точнее ютубовского автоASR — приходит готовая пунктуация и заглавные, так что
шаг `RUPunct` на этом пути отключается сам.

```bash
docker compose up -d pot-provider                # без него не читаются плейлисты
python -m src.etl.pipeline --source asr --drop-non-recipe
```

Прогон резюмируемый: уже обработанные видео пропускаются, так что после обрыва
достаточно перезапустить ту же команду. Полный корпус (666 видео) — около 2.5 ч
на RTX 5060 Ti.

Подводный камень: у роликов канала есть **автодубляж на английский**, и в списке
форматов он идёт первым. Без явного фильтра по языку скачивается он, и Whisper
честно распознаёт английскую речь. Селектор дорожки задан в
`config.ASR_AUDIO_FORMAT` и предпочитает оригинал (`ba[language^=ru]`).

Путь через субтитры сохранён (`--source ytapi|ytdlp|auto`) — пригодится, если
собирать корпус с другого IP или через прокси (`WEBSHARE_PROXY_USERNAME/_PASSWORD`).

## GPU-бюджет (16 ГБ)

vLLM с 14B-AWQ занимает ~14.5 ГБ (веса 9.4 + KV-кэш 5). Поэтому `bge-m3` и
ре-ранкер в рантайме работают на **CPU** (`EMBED_DEVICE=cpu`, `RERANK_DEVICE=cpu`).
Массовое индексирование делается отдельно на GPU при остановленном vLLM.
