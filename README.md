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
src/eval/        Шаг 7   golden-набор → метрики retrieval и генерации
```

Каждый модуль запускается как CLI, напр.:

```bash
python -m src.retrieval.retrieve "маринад для шашлыка"
python -m src.generation.ask "как замариновать курицу" --json
```

## Оценка качества

`data/eval/golden.jsonl` — 38 размеченных запросов: 34 позитивных (ожидаем
конкретные ролики) и 4 негативных (в корпусе рецепта нет, ожидаем честный
`found=false`). Разметка на уровне **видео**, а не чанка: рецепт растянут на
весь ролик, требовать попадания в конкретный чанк бессмысленно.

```bash
python -m src.eval.run --mode retrieval          # быстро, без LLM
python -m src.eval.run --mode both --out report.json
docker compose exec -T api python -m src.eval.run --mode retrieval
```

Путь в `--out` задавайте **относительным** (`--out data/eval/report.json`):
Git Bash разворачивает абсолютный `/app/...` в путь Windows, и отчёт уезжает
мимо контейнера.

| Метрика | Что показывает |
|---|---|
| `hit@k`, `MRR` | нашёлся ли нужный ролик и на каком месте |
| `ctx_precision` | доля пассажей контекста из релевантных роликов — сколько бюджета токенов ушло на посторонние видео |
| `edge_rate` | доля пассажей, начатых крайним чанком (вступление/прощание): там гуще всего звучит название блюда, но рецепта нет |
| `found_acc` | совпал ли `found` с ожиданием — на негативных запросах ловит галлюцинации |
| `avg_steps`, `with_amounts` | детализация ответа и доля ингредиентов с граммовками |

Разбивка по `kind` (`exact` / `paraphrase` / `descriptive`) показывает, где
именно просаживается поиск: на точных названиях, на пересказе или на запросах
без имени блюда.

### Результаты (2026-09-02)

Корпус: 651 ролик, 5239 чанков. Конфигурация: `top_videos=2`, `per_video=6`,
`token_budget=5200`, `rerank_top=24`, эмбеддер и ре-ранкер на CPU.

| Retrieval (34 позитивных) | было | стало | | Генерация (все 38) | было | стало |
|---|---|---|---|---|---|---|
| MRR | 0.941 | **0.956** | | found_acc | 0.868 | **0.947** |
| hit@1 | 0.882 | **0.912** | | attribution | 0.931 | **1.000** |
| hit@3 / hit@5 | 1.000 | 1.000 | | avg_steps | 11.0 | 10.1 |
| ctx_precision | 0.827 | 0.830 | | avg_ingredients | 9.3 | 9.3 |
| edge_rate | 0.120 | **0.108** | | with_amounts | 0.147 | 0.153 |

По типам запроса hit@1: `exact` 0.88, `paraphrase` 1.00,
`descriptive` 0.67 → **0.83**.

Что дало прирост:

- **заголовок ролика в эмбеддируемом тексте** — вытянул `descriptive`, самый
  слабый класс: без него название блюда «видит» только тот чанк, где автор его
  произнёс, то есть вступление;
- **инструкция в промпте про несколько роликов в контексте** — сняла ложные
  отказы. Раньше модель, увидев в контексте два разных блюда, предпочитала
  `found=false`; на этом теряли карбонару, гуляш, роллы и свиные рёбра.

Галлюцинаций нет: на всех четырёх негативных запросах система вернула
`found=false`. `attribution` = 1.0 — когда рецепт найден, ссылка всегда ведёт
на верный ролик.

Оставшиеся два расхождения — «что приготовить из баклажанов» и «блюдо из
кабачков». Ролики находятся, отказывает генерация: вопрос «что приготовить
из X» не про одно блюдо, а схема ответа (`dish` / `ingredients` / `steps`)
рассчитана ровно на одно.

Известные ограничения этого замера: набор мал (34 позитивных запроса — один
переехавший запрос двигает `hit@1` на 3 п.п.), а `hit@3` уже упёрся в 1.000 и
перестал различать варианты.

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
