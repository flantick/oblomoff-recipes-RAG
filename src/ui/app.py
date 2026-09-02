"""Streamlit front-end for the RAG API (Step 6, optional).

    API_URL=http://localhost:8080 streamlit run src/ui/app.py
"""
from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8080")

st.set_page_config(page_title="oblomoff • рецепты", page_icon="🍳", layout="centered")
st.title("🍳 Рецепты oblomoffood")
st.caption("RAG по расшифровкам видео канала. Ответ — только из того, что реально сказано в роликах.")

with st.sidebar:
    st.subheader("Параметры")
    # The UI container ships without src/config.py, so it cannot read RETRIEVAL_*.
    # Hard-coding the numbers here silently overrode the server config on every
    # request — the sliders used to sit at 3/2 while the API had moved to 2/6,
    # and the slider ceiling of 4 made the configured value unreachable anyway.
    # Unless the user opts in, the fields are simply not sent and the API applies
    # its own defaults.
    override = st.checkbox(
        "Задать параметры retrieval вручную",
        value=False,
        help="По умолчанию используются серверные значения RETRIEVAL_* из конфига API",
    )
    top_videos = st.slider("Видео в контексте", 1, 8, 2, disabled=not override)
    per_video = st.slider("Фрагментов на видео", 1, 12, 6, disabled=not override)
    intent = st.checkbox("Фильтр по категории (плейлисту)", value=False)
    try:
        h = requests.get(f"{API_URL}/health", timeout=5).json()
        st.success(f"API: {h['status']}") if h["status"] == "ok" else st.warning(f"API: {h['status']}")
        st.json(h, expanded=False)
    except Exception as exc:  # noqa: BLE001
        st.error(f"API недоступен: {exc}")

query = st.text_input("Что приготовить?", placeholder="как приготовить сочный стейк")

if st.button("Найти рецепт", type="primary") and query:
    with st.spinner("Ищу в видео и собираю рецепт…"):
        try:
            payload = {"query": query, "use_intent_filter": intent}
            if override:
                payload["top_videos"] = top_videos
                payload["per_video"] = per_video
            r = requests.post(f"{API_URL}/ask", json=payload, timeout=180)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Ошибка запроса: {exc}")
            st.stop()

    if not data["found"]:
        st.warning("В расшифровках нет ответа на этот вопрос.")
        for s in data.get("sources", []):
            st.write(f"• [{s['title']}]({s['url']}) — {s['timecode']}")
        st.stop()

    st.header(data["dish"] or query)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Ингредиенты")
        for i in data["ingredients"]:
            st.write(f"- {i}")
    with col2:
        st.subheader("Приготовление")
        for k, step in enumerate(data["steps"], 1):
            st.write(f"{k}. {step}")

    if data.get("notes"):
        st.info(data["notes"])

    src = data.get("source")
    if src:
        st.subheader("Источник")
        st.write(f"**{src['title']}** — таймкод {src['timecode']}")
        st.video(src["url"])

    other = [s for s in data.get("sources", []) if not src or s["n"] != src["n"]]
    if other:
        with st.expander("Ещё фрагменты в контексте"):
            for s in other:
                st.write(f"• [{s['title']}]({s['url']}) — {s['timecode']}")

    st.caption(f"модель: {data.get('model')} · reranker: {data.get('used_reranker')}")
