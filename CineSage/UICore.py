import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
import json
import ast
import uuid
import os

# ============================================================
# SETUP
# ============================================================

load_dotenv()

st.set_page_config(
    page_title="mindHub",
    page_icon="🧠",
    layout="wide"
)

# ============================================================
# LOAD CSS
# ============================================================

def load_css():

    css_path = os.path.join(
        os.path.dirname(__file__),
        "style.css"
    )

    with open(css_path, "r", encoding="utf-8") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

load_css()

# ============================================================
# SESSION STATE
# ============================================================

if "collection" not in st.session_state:
    st.session_state.collection = []

if "current" not in st.session_state:
    st.session_state.current = None

if "history" not in st.session_state:
    st.session_state.history = []

if "linked_blocks" not in st.session_state:
    st.session_state.linked_blocks = []

# ============================================================
# MODEL
# ============================================================

@st.cache_resource
def get_model():

    return ChatMistralAI(
        model="mistral-small-2506",
        api_key=st.secrets["MISTRAL_API_KEY"]
    )

model = get_model()

# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_messages([

    ("system", """
You are an advanced AI assistant — think ChatGPT or Gemini level. You answer any question or topic with depth, clarity, and intelligence.

When the user asks anything — a concept, a coding problem, a how-to, a comparison, a general question, anything — respond with a JSON object with exactly two fields:

1. "plain_text":
   Write a rich, thorough, well-reasoned response exactly like a top AI assistant would.
   - For explanations: cover the concept deeply, give context, examples, real-world relevance
   - For coding questions: explain the approach AND include full working code with comments inside the prose
   - For comparisons: cover pros, cons, use cases in depth
   - For how-to questions: give clear step-by-step reasoning written as prose
   - For opinions or analysis: give a balanced, insightful perspective
   - Length: as long as needed to fully answer — minimum 8 sentences, no upper limit
   - Style: full paras with subtopics and sub-subtopics also whenever needed.
   - Tone: smart, clear, conversational — like a highly knowledgeable friend explaining something properly

2. "subtopics":
   After writing plain_text, extract all key themes or sections from it.
   Each subtopic must:
   - Have a "name" (a natural label for that part of the response)
   - Have "content" taken directly or closely paraphrased from plain_text exactly
   - Cover the full span of the response from beginning to end
   - Contain NO new information not already in plain_text
   - all content of plain_text should be included,no miss of content even the code,examples also

Return ONLY valid JSON. No markdown. No code fences. No extra text outside the JSON structure.
CRITICAL: You must escape all newlines inside strings as \\n and double quotes as \\".
"""),

    ("human", "{query}")

])

# ============================================================
# ASK FUNCTION
# ============================================================

def ask(query: str):

    # ========================================================
    # LINKED MEMORY CONTEXT
    # ========================================================

    linked_context = ""

    if len(st.session_state.linked_blocks) > 0:

        linked_items = [

            item for item in st.session_state.collection

            if item["id"] in st.session_state.linked_blocks
        ]

        if linked_items:

            linked_context += "\n\nLINKED KNOWLEDGE BLOCKS:\n"

            for item in linked_items:

                linked_context += f"""

BLOCK TITLE:
{item["name"]}

BLOCK CONTENT:
{item["content"]}

"""

    enhanced_query = f"""

{linked_context}

USER QUERY:
{query}

Use linked knowledge naturally when relevant.
"""

    final_prompt = prompt.invoke({
        "query": enhanced_query
    })

    response = model.invoke(final_prompt)

    raw = response.content.strip()

    # ========================================================
    # REMOVE MARKDOWN WRAPPERS
    # ========================================================

    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]

    if raw.endswith("```"):
        raw = raw.rsplit("```", 1)[0]

    raw = raw.strip()

    # ========================================================
    # PARSE JSON
    # ========================================================

    try:

        data = json.loads(raw)

    except Exception:

        try:

            data = ast.literal_eval(raw)

        except Exception as e:

            raise ValueError(
                f"Failed parsing response:\n\n{raw}"
            ) from e

    # ========================================================
    # FIX NEWLINES
    # ========================================================

    data["plain_text"] = data.get(
        "plain_text",
        ""
    ).replace("\\n", "\n")

    for s in data.get("subtopics", []):

        s["content"] = s.get(
            "content",
            ""
        ).replace("\\n", "\n")

        s["id"] = "SEM-" + str(uuid.uuid4())[:6]

    return data

# ============================================================
# LAYOUT
# ============================================================

left, middle, right = st.columns(
    [1.1, 2.8, 1.9],
    gap="medium"
)

# ============================================================
# LEFT PANEL
# ============================================================

with left:

    st.markdown(
        f"""
<div class="title-row">
    <div>📚 Collection</div>
    <div class="badge">{len(st.session_state.collection)}</div>
</div>

<div class="subtitle">
    Saved knowledge blocks
</div>
""",
        unsafe_allow_html=True
    )

    left_scroll = st.container(
        height=680,
        border=False
    )

    with left_scroll:

        if len(st.session_state.collection) == 0:

            st.info(
                "No saved blocks yet"
            )

        else:

            for item in st.session_state.collection:

                with st.container(border=True):

                    st.markdown(
                        f"""
            <div class="mini-collection-title">
                {item["name"]}
            </div>
            """,
                        unsafe_allow_html=True
                    )

                    icon1, icon2, icon3 = st.columns(
                        [1,1,1]
                    )

                    # ====================================================
                    # LINK
                    # ====================================================

                    with icon1:

                        linked = (
                            item["id"]
                            in st.session_state.linked_blocks
                        )

                        link_icon = (
                            "🔗"
                            if linked else
                            "⛓️"
                        )

                        if st.button(
                            link_icon,
                            key=f"link_{item['id']}",
                            use_container_width=True
                        ):

                            if linked:

                                st.session_state.linked_blocks.remove(
                                    item["id"]
                                )

                            else:

                                st.session_state.linked_blocks.append(
                                    item["id"]
                                )

                            st.rerun()

                    # ====================================================
                    # VIEW
                    # ====================================================

                    with icon2:

                        if st.button(
                            "👁️",
                            key=f"view_{item['id']}",
                            use_container_width=True
                        ):

                            st.session_state.current = {

                                "query": item["name"],
                                "plain_text": item["content"],
                                "subtopics": [item]

                            }

                            st.rerun()

                    # ====================================================
                    # REMOVE
                    # ====================================================

                    with icon3:

                        if st.button(
                            "🗑️",
                            key=f"remove_{item['id']}",
                            use_container_width=True
                        ):

                            st.session_state.collection = [

                                x for x in st.session_state.collection
                                if x["id"] != item["id"]

                            ]

                            if item["id"] in st.session_state.linked_blocks:

                                st.session_state.linked_blocks.remove(
                                    item["id"]
                                )

                            st.rerun()

# ============================================================
# MIDDLE PANEL
# ============================================================

with middle:

    st.markdown(
        """
<div class="title-row">
    🧠 Knowledge Blocks
</div>

<div class="subtitle">
    Structured AI knowledge
</div>
""",
        unsafe_allow_html=True
    )

    middle_scroll = st.container(
        height=680,
        border=False
    )

    with middle_scroll:

        if st.session_state.current:

            for subtopic in st.session_state.current["subtopics"]:

                with st.container(border=True):

                    st.markdown(
                        f"""
<div class="card-title">
    {subtopic["name"]}
</div>
""",
                        unsafe_allow_html=True
                    )

                    formatted_content = (
                        subtopic["content"]
                        .replace("\n", "<br>")
                    )

                    st.markdown(
                        f"""
<div class="card-content">
    {formatted_content}
</div>
""",
                        unsafe_allow_html=True
                    )

                    already_saved = any(
                        x["id"] == subtopic["id"]
                        for x in st.session_state.collection
                    )

                    if not already_saved:

                        save_clicked = st.button(
                            "📦 Save",
                            key=f"save_{subtopic['id']}"
                        )

                        if save_clicked:

                            st.session_state.collection.append(
                                subtopic
                            )

                            st.rerun()

        else:

            st.info(
                "Ask something to generate knowledge blocks"
            )

# ============================================================
# RIGHT PANEL
# ============================================================

with right:

    st.markdown(
        """
<div class="title-row">
    💬 Ask
</div>

<div class="subtitle">
    Ask anything
</div>
""",
        unsafe_allow_html=True
    )

    # ========================================================
    # ACTIVE LINKED CONTEXT
    # ========================================================

    if len(st.session_state.linked_blocks) > 0:

        linked_names = [

            item["name"]

            for item in st.session_state.collection

            if item["id"] in st.session_state.linked_blocks
        ]

        st.markdown(
            f"""
<div class="linked-memory-box">
    🔗 Linked Context:<br>
    {", ".join(linked_names)}
</div>
""",
            unsafe_allow_html=True
        )

    chat_height = 470

    if len(st.session_state.linked_blocks) > 0:
        chat_height = 400

    chat_scroll = st.container(
        height=chat_height,
        border=False
    )

    with chat_scroll:

        if st.session_state.current:

            st.markdown(
                f"""
<div class="query-box">
    {st.session_state.current["query"]}
</div>
""",
                unsafe_allow_html=True
            )

            formatted_response = (
                st.session_state.current["plain_text"]
                .replace("\n", "<br>")
            )

            st.markdown(
                f"""
<div class="chat-response">
    {formatted_response}
</div>
""",
                unsafe_allow_html=True
            )

        else:

            st.info(
                "No response yet"
            )

    user_input = st.text_area(
        "Ask",
        label_visibility="collapsed",
        placeholder="Ask anything...",
        height=95
    )

    ask_clicked = st.button(
        "⚡ Ask",
        use_container_width=True
    )

    if ask_clicked and user_input.strip():

        with st.spinner("Generating response..."):

            try:

                result = ask(user_input)

                st.session_state.current = {

                    "query": user_input,
                    "plain_text": result["plain_text"],
                    "subtopics": result["subtopics"]

                }

                st.session_state.history.append(
                    st.session_state.current
                )

                st.rerun()

            except Exception as e:

                st.error(str(e))