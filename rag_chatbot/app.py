"""
RAG-Based Custom Chatbot for Company Data (Gemini version)
============================================================
Loads company data (.txt and .xlsx files) from a local `data/` folder at
startup, chunks + indexes it into a FAISS vector store, and answers
employee questions using only the retrieved content via Google's free
Gemini API (LLM + embeddings).

Run with:
    streamlit run app.py
"""

import os
import glob
from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader, UnstructuredExcelLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
FAISS_INDEX_DIR = os.path.join(os.path.dirname(__file__), "faiss_index")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
EMBEDDING_MODEL = "models/gemini-embedding-001"
# Primary chat model, with fallbacks tried in order if the primary 404s
# (Google retires/renames models fairly often — keep this list current).
CHAT_MODEL_CANDIDATES = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
]
TOP_K = 4

st.set_page_config(page_title="Company Knowledge Assistant", page_icon="🤖", layout="wide")

# Load external CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Prompt — forces the LLM to answer ONLY from retrieved context
# --------------------------------------------------------------------------
QA_PROMPT = PromptTemplate(
    template="""You are an internal company assistant. Answer the employee's
question using ONLY the context provided below, which was retrieved from the
company's own data files. Do not use any outside knowledge.

If the answer is not contained in the context, respond exactly with:
"I don't have that information in the company documents I was given."

Always be concise and professional. When helpful, mention which topic/file
area the information came from.

Context:
---------
{context}
---------

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


# --------------------------------------------------------------------------
# Data loading + indexing (cached so it only runs once per session/app life)
# --------------------------------------------------------------------------
def load_documents(data_dir: str):
    """Load all .txt and .xlsx files from the data directory."""
    documents = []

    txt_files = glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True)
    for path in txt_files:
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = os.path.basename(path)
            d.metadata["file_size"] = os.path.getsize(path)
        documents.extend(docs)

    xlsx_files = glob.glob(os.path.join(data_dir, "**", "*.xlsx"), recursive=True)
    for path in xlsx_files:
        loader = UnstructuredExcelLoader(path, mode="elements")
        docs = loader.load()
        for d in docs:
            d.metadata["source_file"] = os.path.basename(path)
            d.metadata["file_size"] = os.path.getsize(path)
        documents.extend(docs)

    return documents


@st.cache_resource(show_spinner=False)
def build_vectorstore(_api_key: str):
    """Load, chunk, embed, and index all company data. Cached across reruns."""
    documents = load_documents(DATA_DIR)

    if not documents:
        return None, 0, 0, datetime.now()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        return None, len(documents), 0, datetime.now()

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return vectorstore, len(documents), len(chunks), datetime.now()


@st.cache_resource(show_spinner=False)
def pick_working_chat_model(_api_key: str):
    """
    Try each candidate model with a trivial call and return the name of the
    first one that actually works for this API key, instead of hardcoding
    a single model name that might get deprecated later.
    """
    for model_name in CHAT_MODEL_CANDIDATES:
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                google_api_key=_api_key,
                timeout=20,
            )
            llm.invoke("ping")
            return model_name
        except Exception:
            continue
    # Fall back to the first candidate; the real error will surface later
    # with a clear message rather than failing silently here.
    return CHAT_MODEL_CANDIDATES[0]


def build_qa_chain(vectorstore, api_key: str, chat_model: str):
    llm = ChatGoogleGenerativeAI(
        model=chat_model,
        temperature=0,
        google_api_key=api_key,
        streaming=True,
        timeout=60,  # 'timeout', not 'request_timeout', on langchain-google-genai >=4.0
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["query"])),
            question=lambda x: x["query"],
        )
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain, retriever


def format_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.1f} MB"


# --------------------------------------------------------------------------
# Sidebar — status, config, file list
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-section-title">⚙️ Setup</div>', unsafe_allow_html=True)

    api_key = os.getenv("GOOGLE_API_KEY", "")
    api_key_input = st.text_input(
        "Google (Gemini) API Key",
        value=api_key,
        type="password",
        help="Free key: https://aistudio.google.com/app/apikey",
    )

    st.markdown('<div class="sidebar-section-title">📚 Knowledge Base</div>', unsafe_allow_html=True)
    
    # Calculate list of unique files and sizes for sidebar display
    all_txt = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    all_xlsx = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    all_files = sorted(all_txt + all_xlsx)
    
    if all_files:
        for f in all_files:
            bname = os.path.basename(f)
            ext = os.path.splitext(bname)[1].lower()
            icon = "📄" if ext == ".txt" else "📊"
            size = format_size(os.path.getsize(f))
            st.markdown(f'<div class="file-item"><span>{icon}</span><span style="flex-grow:1; font-weight:500;">{bname}</span><span style="opacity:0.6; font-size:0.75rem;">{size}</span></div>', unsafe_allow_html=True)
    else:
        st.warning("No files found in `data/`.")

    if 'vectorstore_info' in st.session_state:
        num_docs, num_chunks, last_built = st.session_state.vectorstore_info
        st.markdown(f'''
        <div class="health-metric" style="margin-top: 1rem;">
            <div class="health-metric-title">Index Health</div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.8rem;">Documents:</span>
                <span class="health-metric-value">{num_docs}</span>
            </div>
            <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                <span style="font-size:0.8rem;">Chunks:</span>
                <span class="health-metric-value">{num_chunks}</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span style="font-size:0.8rem;">Last Built:</span>
                <span style="font-size:0.8rem; font-weight:500;">{last_built.strftime("%H:%M:%S")}</span>
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">⚡ Actions</div>', unsafe_allow_html=True)
    
    if st.button("🔄 Rebuild index", type="primary", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pop("qa_chain", None)
        st.session_state.pop("vectorstore_info", None)
        st.toast("Index cache cleared. Rebuilding on next run...", icon="🔄")
        st.rerun()

    if st.button("🗑️ Clear chat history", type="secondary", use_container_width=True):
        st.session_state["messages"] = []
        st.toast("Chat history cleared!", icon="✅")
        st.rerun()

# --------------------------------------------------------------------------
# Main app
# --------------------------------------------------------------------------
# Custom Header
st.markdown("""
<div class="app-header">
    <div class="app-logo">🤖</div>
    <div class="app-title-wrapper">
        <h1>Company Knowledge</h1>
        <p class="app-subtitle">Internal tool powered by Gemini RAG</p>
    </div>
</div>
""", unsafe_allow_html=True)

if not api_key_input:
    st.info(
        "👈 Enter your Google (Gemini) API key in the sidebar to get started. "
        "You can get a free key at https://aistudio.google.com/app/apikey"
    )
    st.stop()

with st.spinner("Indexing company data..."):
    vectorstore, num_docs, num_chunks, last_built = build_vectorstore(api_key_input)
    st.session_state.vectorstore_info = (num_docs, num_chunks, last_built)

if vectorstore is None:
    st.error(
        f"No documents were found in `{DATA_DIR}`. Add .txt or .xlsx files there and click "
        "'Rebuild index' in the sidebar."
    )
    st.stop()

if "chat_model" not in st.session_state:
    with st.spinner("Checking model availability..."):
        st.session_state.chat_model = pick_working_chat_model(api_key_input)

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain, st.session_state.retriever = build_qa_chain(
        vectorstore, api_key_input, st.session_state.chat_model
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

# Empty State
if len(st.session_state.messages) == 0:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 1rem;">
        <h3 style="margin-bottom:0.5rem; font-weight:600;">Welcome to Company Knowledge!</h3>
        <p style="color:var(--text-color); opacity:0.7; margin-bottom: 2rem;">
            Ask questions about company policies, FAQs, and internal data. <br/>
            Answers are strictly grounded in our internal documents.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("**Try asking:**")
    col1, col2, col3 = st.columns(3)
    suggestions = [
        "What is the company vacation policy?",
        "How do I expense a work meal?",
        "What are our core values?"
    ]
    for idx, (col, text) in enumerate(zip([col1, col2, col3], suggestions)):
        with col:
            st.markdown('<div class="suggestion-btn">', unsafe_allow_html=True)
            if st.button(text, key=f"sugg_{idx}"):
                # If clicked, append to chat directly
                st.session_state.selected_suggestion = text
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(f'<span class="chat-timestamp">{msg.get("timestamp", "")}</span>', unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sources"):
            # Render sources as pill tags
            pills_html = '<div class="source-pill-container">'
            for s in msg["sources"]:
                ext = os.path.splitext(s)[1].lower()
                icon = "📄" if ext == ".txt" else "📊"
                pills_html += f'<span class="source-pill">{icon} {s}</span>'
            pills_html += '</div>'
            st.markdown(pills_html, unsafe_allow_html=True)


# Determine what question to run (either from suggestion or input)
run_question = None
if 'selected_suggestion' in st.session_state:
    run_question = st.session_state.selected_suggestion
    del st.session_state.selected_suggestion

# Chat input
question_input = st.chat_input("Ask a question about company data...")

if question_input:
    run_question = question_input

if run_question:
    timestamp = datetime.now().strftime("%I:%M %p")
    st.session_state.messages.append({"role": "user", "content": run_question, "timestamp": timestamp})
    
    with st.chat_message("user"):
        st.markdown(f'<span class="chat-timestamp">{timestamp}</span>', unsafe_allow_html=True)
        st.markdown(run_question)

    with st.chat_message("assistant"):
        answer = ""
        sources = []
        try:
            with st.spinner("Searching knowledge base..."):
                source_docs = st.session_state.retriever.invoke(run_question)
                sources = sorted(
                    {doc.metadata.get("source_file", "unknown") for doc in source_docs}
                )

            # Generate and stream the answer
            def token_stream():
                for chunk in st.session_state.qa_chain.stream({"query": run_question}):
                    yield chunk

            answer = st.write_stream(token_stream())

        except Exception as e:
            import traceback
            err_detail = traceback.format_exc()
            answer = f"⚠️ An error occurred: {e}"
            st.error(answer)
            st.code(err_detail, language="text")

        timestamp = datetime.now().strftime("%I:%M %p")
        st.markdown(f'<span class="chat-timestamp">{timestamp}</span>', unsafe_allow_html=True)
        
        if sources:
            pills_html = '<div class="source-pill-container">'
            for s in sources:
                ext = os.path.splitext(s)[1].lower()
                icon = "📄" if ext == ".txt" else "📊"
                pills_html += f'<span class="source-pill">{icon} {s}</span>'
            pills_html += '</div>'
            st.markdown(pills_html, unsafe_allow_html=True)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "timestamp": timestamp}
    )

st.markdown('<div class="internal-footer">For internal company use only. Responses are generated by AI based on loaded company documents. Using model: {}</div>'.format(st.session_state.get('chat_model', 'unknown')), unsafe_allow_html=True)
