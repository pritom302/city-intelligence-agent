import html
import json
import os
import re
import uuid

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from city_tools import get_news, get_weather

# ========================================
# PAGE CONFIG + STYLES
# ========================================

st.set_page_config(
    page_title="City Intelligence",
    page_icon="🌆",
    layout="centered",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
.block-container { padding-top: 1.5rem; max-width: 960px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- Hero ---------- */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #ec4899 100%);
    border-radius: 22px;
    padding: 2rem 2.2rem;
    color: #fff;
    margin-bottom: 1.4rem;
    box-shadow: 0 12px 40px rgba(124, 58, 237, 0.35);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "🌆";
    position: absolute; right: 24px; top: 6px;
    font-size: 6.5rem; opacity: 0.16;
}
.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.20);
    border: 1px solid rgba(255,255,255,0.35);
    padding: 3px 12px; border-radius: 999px;
    font-size: 0.72rem; letter-spacing: 0.08em; font-weight: 600;
    margin-bottom: 0.7rem;
}
.hero-title { font-size: 2.3rem; font-weight: 800; line-height: 1.1; margin: 0; }
.hero-sub { margin: 0.5rem 0 0 0; font-size: 1.02rem; opacity: 0.92; max-width: 560px; }

/* ---------- Buttons ---------- */
div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
    width: 100%;
    border-radius: 14px;
    padding: 0.65rem 1rem;
    border: 1px solid rgba(124, 58, 237, 0.35);
    transition: all .18s ease;
    font-weight: 600;
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover {
    border-color: #7c3aed;
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(124, 58, 237, 0.25);
}

/* ---------- Weather card ---------- */
.wx-card {
    border-radius: 22px; padding: 1.5rem 1.7rem; color: #fff;
    margin: 0.2rem 0 1rem 0;
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}
.wx-default { background: linear-gradient(135deg, #6366f1, #8b5cf6); }
.wx-clear   { background: linear-gradient(135deg, #f59e0b, #ef4444); }
.wx-cloud   { background: linear-gradient(135deg, #64748b, #334155); }
.wx-rain    { background: linear-gradient(135deg, #2563eb, #1e3a8a); }
.wx-storm   { background: linear-gradient(135deg, #4c1d95, #1e1b4b); }
.wx-snow    { background: linear-gradient(135deg, #38bdf8, #6366f1); }
.wx-mist    { background: linear-gradient(135deg, #94a3b8, #64748b); }
.wx-top { display: flex; justify-content: space-between; align-items: flex-start; }
.wx-city { font-size: 1.25rem; font-weight: 700; }
.wx-cond { font-size: 0.98rem; opacity: 0.92; text-transform: capitalize; }
.wx-icon { font-size: 3.4rem; line-height: 1; }
.wx-temp { font-size: 4rem; font-weight: 800; line-height: 1; margin: 0.4rem 0 1rem 0; }
.wx-temp span { font-size: 1.6rem; font-weight: 600; opacity: 0.85; margin-left: 4px; }
.wx-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.7rem; }
.wx-stat {
    background: rgba(255,255,255,0.17); border-radius: 14px;
    padding: 0.65rem 0.8rem; backdrop-filter: blur(6px);
}
.wx-stat-label { font-size: 0.72rem; opacity: 0.85; text-transform: uppercase; letter-spacing: .06em; }
.wx-stat-value { font-size: 1.12rem; font-weight: 700; margin-top: 2px; }

/* ---------- News cards ---------- */
.news-head { font-weight: 700; margin: 0.3rem 0 0.5rem 0; font-size: 1.05rem; }
.news-card {
    border-left: 4px solid #7c3aed;
    background: rgba(124, 58, 237, 0.07);
    border-radius: 12px; padding: 0.75rem 1rem; margin-bottom: 0.6rem;
}
.news-card a { font-weight: 700; text-decoration: none; color: #7c3aed; }
.news-card a:hover { text-decoration: underline; }
.news-snippet { font-size: 0.9rem; opacity: 0.85; margin-top: 4px; }
.news-src { font-size: 0.75rem; opacity: 0.6; margin-top: 4px; }

/* ---------- Sidebar chips ---------- */
.chip {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: 0.78rem; font-weight: 600; margin: 0 6px 6px 0;
}
.chip-ok  { background: rgba(34,197,94,0.15);  color: #16a34a; border: 1px solid rgba(34,197,94,0.4); }
.chip-bad { background: rgba(239,68,68,0.13);  color: #dc2626; border: 1px solid rgba(239,68,68,0.4); }
.side-title { font-weight: 800; font-size: 1.15rem; margin-bottom: 0.4rem; }

/* ---------- Approval box ---------- */
.approve-title { font-weight: 800; font-size: 1.05rem; margin-bottom: 0.2rem; }
.approve-sub { opacity: 0.75; font-size: 0.9rem; margin-bottom: 0.6rem; }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ========================================
# CONSTANTS
# ========================================

SYSTEM_PROMPT = """
You are a City Intelligence Assistant.

You have two tools:
1. get_weather(city): Gets current weather information.
2. get_news(city): Searches for recent news and local updates.

Rules:
- For current weather questions, use get_weather.
- For latest news questions, use get_news.
- For requests about both weather and news, use both tools.
- Pass the location provided by the user to the tools.
- For locations in Bangladesh, include the district and country
  when needed to distinguish places with similar names.
- The app already shows weather and news as cards, so reply with a short,
  friendly summary instead of repeating every number or headline.
- Never invent weather data or news.
- If a tool is denied or fails, explain that to the user.
"""

TOOL_META = {
    "get_weather": {"icon": "🌤️", "label": "Weather lookup"},
    "get_news": {"icon": "📰", "label": "News search"},
}

QUICK_PROMPTS = [
    ("☀️  Weather in Habiganj", "What's the weather in Habiganj, Bangladesh?"),
    ("📰  News in Sylhet", "What's the latest news in Sylhet, Bangladesh?"),
    ("🌆  Weather + news for Dhaka", "Give me the weather and latest news for Dhaka."),
    ("🌧️  Is it raining in Chittagong?", "Is it raining in Chittagong right now?"),
]

MODEL_NAME = "codestral-latest"

# ========================================
# AGENT
# ========================================


@st.cache_resource(show_spinner=False)
def get_checkpointer():
    """One shared memory store so chats survive Streamlit reruns."""
    return InMemorySaver()


@st.cache_resource(show_spinner=False)
def build_agent(model_name: str, require_approval: bool):
    llm = ChatMistralAI(
        model=model_name,
        temperature=0,
        max_retries=1,
        timeout=60,
    )

    middleware = []
    if require_approval:
        middleware.append(
            HumanInTheLoopMiddleware(
                interrupt_on={"get_weather": True, "get_news": True}
            )
        )

    return create_agent(
        llm,
        tools=[get_weather, get_news],
        system_prompt=SYSTEM_PROMPT,
        middleware=middleware,
        checkpointer=get_checkpointer(),
    )


def invoke_agent(payload):
    agent = build_agent(MODEL_NAME, st.session_state.approval)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    return agent.invoke(payload, config)


# ========================================
# HELPERS
# ========================================


def init_state():
    defaults = {
        "messages": [],
        "pending": None,
        "thread_id": str(uuid.uuid4()),
        "queued": None,
        "round": 0,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def friendly_error(e: Exception) -> str:
    s = str(e)
    if "429" in s or "rate limit" in s.lower():
        return (
            "⚠️ Mistral rate limit reached. The free plan allows only a few "
            "requests per minute, so wait about a minute and try again."
        )
    return f"⚠️ Something went wrong: {s}"


def weather_style(condition: str):
    c = condition.lower()
    if "thunder" in c:
        return "⛈️", "storm"
    if "rain" in c or "drizzle" in c:
        return "🌧️", "rain"
    if "snow" in c:
        return "❄️", "snow"
    if any(w in c for w in ("mist", "fog", "haze", "smoke", "dust")):
        return "🌫️", "mist"
    if "cloud" in c or "overcast" in c:
        return "☁️", "cloud"
    if "clear" in c:
        return "☀️", "clear"
    return "🌤️", "default"


def parse_weather(text: str):
    if not text or "Temperature:" not in text:
        return None
    data = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip().lower()] = value.strip()
    return data


def render_weather_card(raw: str):
    data = parse_weather(raw)
    if not data:
        return

    def num(value: str) -> str:
        try:
            return f"{float(value.split(' ')[0]):.0f}"
        except (ValueError, IndexError):
            return html.escape(value)

    condition = data.get("condition", "")
    icon, theme = weather_style(condition)

    card = f"""
    <div class="wx-card wx-{theme}">
      <div class="wx-top">
        <div>
          <div class="wx-city">📍 {html.escape(data.get("city", ""))}</div>
          <div class="wx-cond">{html.escape(condition)}</div>
        </div>
        <div class="wx-icon">{icon}</div>
      </div>
      <div class="wx-temp">{num(data.get("temperature", "?"))}°<span>C</span></div>
      <div class="wx-grid">
        <div class="wx-stat">
          <div class="wx-stat-label">Feels like</div>
          <div class="wx-stat-value">{num(data.get("feels like", "?"))}°C</div>
        </div>
        <div class="wx-stat">
          <div class="wx-stat-label">Humidity</div>
          <div class="wx-stat-value">{html.escape(data.get("humidity", "?"))}</div>
        </div>
        <div class="wx-stat">
          <div class="wx-stat-label">Wind</div>
          <div class="wx-stat-value">{html.escape(data.get("wind speed", "?"))}</div>
        </div>
      </div>
    </div>
    """
    st.markdown(card, unsafe_allow_html=True)


def render_news_cards(raw: str):
    items = re.findall(
        r"Title: (.*?)\nURL: (.*?)\nContent: (.*?)(?=\n\nTitle: |\Z)",
        raw,
        re.S,
    )
    if not items:
        return

    st.markdown('<div class="news-head">📰 Latest news</div>', unsafe_allow_html=True)
    for title, url, content in items[:5]:
        snippet = content.strip().replace("\n", " ")
        if len(snippet) > 180:
            snippet = snippet[:180].rsplit(" ", 1)[0] + "…"
        domain = re.sub(r"^https?://(www\.)?", "", url.strip()).split("/")[0]
        st.markdown(
            f"""
            <div class="news-card">
              <a href="{html.escape(url.strip())}" target="_blank">{html.escape(title.strip())}</a>
              <div class="news-snippet">{html.escape(snippet)}</div>
              <div class="news-src">🔗 {html.escape(domain)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_tool_cards(tools):
    for t in tools:
        if t["status"] != "ok" or not t.get("output"):
            continue
        if t["name"] == "get_weather":
            render_weather_card(t["output"])
        elif t["name"] == "get_news":
            render_news_cards(t["output"])


def render_tool_activity(tools):
    with st.expander(f"🔧 Tool activity ({len(tools)})"):
        for t in tools:
            meta = TOOL_META.get(t["name"], {"icon": "🔧", "label": t["name"]})
            badge = {"ok": "✅ completed", "denied": "🚫 denied / failed"}.get(
                t["status"], "⏳ no result"
            )
            st.markdown(f"**{meta['icon']} {meta['label']}** · `{t['name']}` · {badge}")
            st.json(t["args"])
            if t.get("output"):
                st.code(t["output"], language=None)


def render_message(msg):
    avatar = "🧑" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            render_tool_cards(msg.get("tools", []))
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("tools"):
            render_tool_activity(msg["tools"])


def collect_turn(messages):
    """Pull the final answer and tool results out of the latest turn."""
    last_human = max(
        (i for i, m in enumerate(messages) if isinstance(m, HumanMessage)),
        default=-1,
    )
    turn = messages[last_human + 1:]

    calls = {}
    for m in turn:
        if isinstance(m, AIMessage) and m.tool_calls:
            for c in m.tool_calls:
                calls[c["id"]] = {
                    "name": c["name"],
                    "args": c["args"],
                    "output": None,
                    "status": "pending",
                }
        elif isinstance(m, ToolMessage):
            entry = calls.get(m.tool_call_id)
            if entry:
                output = text_of(m.content)
                failed = getattr(m, "status", None) == "error" or output.startswith(
                    ("Error", "Weather", "News search error")
                )
                entry["output"] = output
                entry["status"] = "denied" if failed else "ok"

    answer = ""
    for m in reversed(turn):
        if isinstance(m, AIMessage) and not m.tool_calls:
            answer = text_of(m.content)
            break

    return answer or "I couldn't produce an answer. Please try again.", list(calls.values())


def handle_result(result):
    interrupts = result.get("__interrupt__")
    if interrupts:
        first = interrupts[0]
        st.session_state.pending = first.value if hasattr(first, "value") else first
        st.session_state.round += 1
        return

    st.session_state.pending = None
    answer, tools = collect_turn(result["messages"])
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "tools": tools}
    )


def add_error(e: Exception):
    st.session_state.pending = None
    st.session_state.messages.append(
        {"role": "assistant", "content": friendly_error(e), "tools": []}
    )


# ========================================
# SIDEBAR
# ========================================

init_state()

with st.sidebar:
    st.markdown('<div class="side-title">⚙️ Settings</div>', unsafe_allow_html=True)

    st.toggle(
        "Ask before running tools",
        value=True,
        key="approval",
        help="When on, you approve each weather or news lookup before it runs.",
    )

    st.markdown("---")
    st.markdown("**API keys**")
    chips = ""
    for label, env in [
        ("Mistral", "MISTRAL_API_KEY"),
        ("OpenWeather", "OPENWEATHER_API_KEY"),
        ("Tavily", "TAVILY_API_KEY"),
    ]:
        ok = bool(os.getenv(env))
        css = "chip-ok" if ok else "chip-bad"
        mark = "✓" if ok else "✕"
        chips += f'<span class="chip {css}">{mark} {label}</span>'
    st.markdown(chips, unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Tip: the free Mistral plan allows only a few requests per minute. "
        "If you see a rate limit message, wait a minute."
    )

    if st.button("🗑️  New chat"):
        st.session_state.messages = []
        st.session_state.pending = None
        st.session_state.queued = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

# ========================================
# MAIN PAGE
# ========================================

st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">LIVE WEATHER · LOCAL NEWS</div>
      <div class="hero-title">City Intelligence</div>
      <p class="hero-sub">Ask about the weather and the latest news for any city.
      An AI agent picks the right tools and shows you the results.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Empty state: quick prompts
if not st.session_state.messages and not st.session_state.pending:
    st.markdown("##### ✨ Try asking")
    cols = st.columns(2)
    for i, (label, question) in enumerate(QUICK_PROMPTS):
        if cols[i % 2].button(label, key=f"qp_{i}"):
            st.session_state.queued = question
            st.rerun()

# Chat history
for msg in st.session_state.messages:
    render_message(msg)

# Pending tool approval
pending = st.session_state.pending
if pending:
    requests_ = pending.get("action_requests", [])
    with st.chat_message("assistant", avatar="🛡️"):
        st.markdown(
            '<div class="approve-title">Approval needed</div>'
            '<div class="approve-sub">The agent wants to run the tools below. '
            "Untick any you want to block.</div>",
            unsafe_allow_html=True,
        )
        with st.form(f"approval_form_{st.session_state.round}"):
            allowed = []
            for i, req in enumerate(requests_):
                meta = TOOL_META.get(req["name"], {"icon": "🔧", "label": req["name"]})
                args = ", ".join(f"{k}={v!r}" for k, v in req.get("args", {}).items())
                allowed.append(
                    st.checkbox(
                        f"{meta['icon']} **{meta['label']}** — `{req['name']}({args})`",
                        value=True,
                        key=f"allow_{st.session_state.round}_{i}",
                    )
                )
            submitted = st.form_submit_button("Continue ▶", type="primary")

    if submitted:
        decisions = [
            {"type": "approve"}
            if ok
            else {"type": "reject", "message": "Tool call denied by the user."}
            for ok in allowed
        ]
        with st.spinner("Working on it…"):
            try:
                result = invoke_agent(Command(resume={"decisions": decisions}))
                handle_result(result)
            except Exception as e:
                add_error(e)
        st.rerun()

# Chat input
prompt = st.chat_input(
    "Ask about weather or news in any city…",
    disabled=bool(st.session_state.pending),
)

if st.session_state.queued and not st.session_state.pending:
    prompt = st.session_state.queued
    st.session_state.queued = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            try:
                result = invoke_agent(
                    {"messages": [{"role": "user", "content": prompt}]}
                )
                handle_result(result)
            except Exception as e:
                add_error(e)
    st.rerun()