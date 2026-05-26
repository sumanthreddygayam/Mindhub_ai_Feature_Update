import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import json
import ast
import uuid
import os

load_dotenv()

st.set_page_config(
    page_title="mindHub Clone",
    page_icon="🧠",
    layout="wide"
)

# --- CSS Loader ---
def load_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"CSS file '{file_name}' not found.")

load_css("style.css")

# --- State Initialization ---
if "collection" not in st.session_state:
    st.session_state.collection = []
if "generated_data" not in st.session_state:
    st.session_state.generated_data = None
if "show_collection" not in st.session_state:
    st.session_state.show_collection = False

# -------------------- Model & Prompt --------------------
@st.cache_resource
def get_model():
    return ChatMistralAI(model="mistral-small-2506")

model = get_model()

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an advanced semantic topic decomposition AI.
Break the user's topic into 2-3 meaningful subtopics.
Return ONLY valid JSON.
Output format:
{{
    "main_topic": "...",
    "subtopics": [
        {{"name": "...", "content": "..."}}
    ]
}}
"""),
    ("human", "{topic}")
])

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sb-logo">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"/></svg>
        <span style="font-weight:700;">mind</span>Hub
    </div>
    
    <div style="padding: 0 10px 10px;">
        <div style="padding:8px 12px; border-radius:8px; border:1px solid #EAEAEA; background:#fff; cursor:pointer; font-size:13px; font-weight:500;">+ New Thought map</div>
    </div>

    <div class="sb-nav-item">🔍 Search <span style="font-size:10px;color:#999;margin-left:auto;border:1px solid #EAEAEA;padding:2px 4px;border-radius:4px;">Ctrl K</span></div>
    <div class="sb-nav-item" style="color:#aaa;">📄 Articles <span style="font-size:9px;background:#F0F0F0;padding:2px 4px;border-radius:4px;margin-left:auto;">SOON</span></div>
    <div class="sb-nav-item" style="color:#aaa;">🤖 Agents <span style="font-size:9px;background:#F0F0F0;padding:2px 4px;border-radius:4px;margin-left:auto;">SOON</span></div>
    <div class="sb-nav-item">📝 Prompts</div>
    <div class="sb-nav-item">📚 Knowledge Base</div>

    <div class="sb-section">Yesterday</div>
    <div class="sb-history-item active">DFS Graph</div>

    <div class="sb-section">Older</div>
    <div class="sb-history-item">Greeting</div>
    <div class="sb-history-item">Sustainability</div>
    <div class="sb-history-item">Newton LSTM</div>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN LAYOUT
# =========================================================
# Using gap="large" prevents overlapping
mid_col, right_col = st.columns([55, 45], gap="large")

# ─────────────────────────────────────────
# MIDDLE CANVAS
# ─────────────────────────────────────────
with mid_col:
    # Wrap the middle content in the dotted background container
    st.markdown('<div class="canvas-wrapper">', unsafe_allow_html=True)
    
    # Safe Toggle Button for Saved Blocks
    col1, col2 = st.columns([3, 7])
    with col1:
        if st.button(f"📄 Notes/Files ({len(st.session_state.collection)})", key="toggle_coll", use_container_width=True):
            st.session_state.show_collection = not st.session_state.show_collection

    if st.session_state.show_collection:
        st.markdown('<div class="coll-modal">', unsafe_allow_html=True)
        st.markdown('<div class="coll-modal-header">Saved Blocks</div>', unsafe_allow_html=True)
        if len(st.session_state.collection) == 0:
            st.markdown('<div style="padding:16px; font-size:12px; color:#888; text-align:center;">Empty</div>', unsafe_allow_html=True)
        else:
            for item in st.session_state.collection:
                cc1, cc2 = st.columns([5, 1])
                with cc1:
                    st.markdown(f'<div class="coll-block-item">{item["name"]}</div>', unsafe_allow_html=True)
                with cc2:
                    if st.button("✕", key=f"del_{item['id']}"):
                        st.session_state.collection = [x for x in st.session_state.collection if x["id"] != item["id"]]
                        st.rerun()
                st.markdown('<hr style="margin:0; border:none; border-top:1px solid #F5F5F5;">', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<h2 style="margin-top:30px; font-weight:700;">Chats</h2>', unsafe_allow_html=True)

    # Render Cards
    if st.session_state.generated_data:
        data = st.session_state.generated_data
        st.markdown(f'<div style="font-weight:500; font-size:16px; margin-bottom:16px;">{data["main_topic"]}<br><span style="font-size:12px;color:#888;">May 25, 10:24 PM</span></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="kcard-wrapper">', unsafe_allow_html=True)
        for subtopic in data["subtopics"]:
            # Clean sentences for bullets
            sentences = [s.strip() for s in subtopic["content"].split(". ") if s.strip()]
            bullets_html = "".join([f'<div class="kcard-bullet">{s}</div>' for s in sentences[:3]])
            
            st.markdown(f"""
            <div class="kcard">
                <div class="kcard-header">{subtopic['name'].lower()}</div>
                <div class="kcard-body">{bullets_html}</div>
            </div>
            """, unsafe_allow_html=True)
            
            already_saved = any(x["id"] == subtopic["id"] for x in st.session_state.collection)
            if already_saved:
                st.markdown('<div style="margin-top: -10px; margin-bottom: 20px;"><span style="color:#5C2B1D; font-size:18px;">★</span> Saved</div>', unsafe_allow_html=True)
            else:
                if st.button("☆ Save to Notes", key=f"save_{subtopic['id']}"):
                    st.session_state.collection.append(subtopic)
                    st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True) # End Canvas Wrapper

# ─────────────────────────────────────────
# RIGHT PANEL
# ─────────────────────────────────────────
with right_col:
    # Adding a top margin container so it aligns visually with the middle section
    st.markdown('<div style="padding-top: 20px;">', unsafe_allow_html=True)

    if st.session_state.generated_data:
        data = st.session_state.generated_data
        st.markdown(f"""
        <div class="rp-header">
            <div>
                <span class="rp-title">{data['main_topic']}</span>
                <span class="rp-date">May 25, 10:24 PM</span>
            </div>
            <div class="rp-share">Share</div>
        </div>
        <div class="rp-section-title">Knowledge Template</div>
        """, unsafe_allow_html=True)

        for subtopic in data["subtopics"]:
            st.markdown(f"""
            <div style="font-size:14px; margin-bottom:6px;">{subtopic['name']}:</div>
            <div class="rp-block">
                <div class="rp-block-header">
                    <span>&lt;/&gt; Text</span>
                    <span style="font-size:14px; cursor:pointer;">📥</span>
                </div>
                <div class="rp-block-content">{subtopic['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="rp-header">
            <div>
                <span class="rp-title">Explore Topic</span>
            </div>
            <div class="rp-share">Share</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Input Area ──
    st.markdown('<div style="margin-top: 40px;">', unsafe_allow_html=True)
    topic = st.text_area("Input", height=100, placeholder="What are we exploring today?", label_visibility="hidden")
    
    col_icn, col_gen = st.columns([7, 3])
    with col_icn:
        st.markdown('<div style="font-size:18px; color:#888; padding-top:10px; letter-spacing:10px;">＋ 💡 ↻ 📖</div>', unsafe_allow_html=True)
    with col_gen:
        gen = st.button("Generate", use_container_width=True)

    st.markdown('<div style="text-align:center; font-size:11px; color:#aaa; margin-top:12px;">AI models on mindHub can make mistakes. Check important info.</div>', unsafe_allow_html=True)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # Generation Logic
    if gen and topic.strip():
        with st.spinner("Thinking..."):
            try:
                final_prompt = prompt.invoke({"topic": topic})
                response = model.invoke(final_prompt)
                raw_output = response.content.strip().replace("```json", "").replace("```", "").strip()
                try:
                    data = json.loads(raw_output)
                except:
                    data = ast.literal_eval(raw_output)

                for subtopic in data["subtopics"]:
                    subtopic["id"] = "blk_" + str(uuid.uuid4())[:6]

                st.session_state.generated_data = data
                st.rerun()
            except Exception as e:
                st.error("Error formatting JSON. Try again.")