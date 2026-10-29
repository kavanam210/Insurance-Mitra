"""Insurance Mitra: a session-aware, evidence-grounded insurance explainer."""

from __future__ import annotations

import os

import streamlit as st
from google.genai.errors import APIError
from sentence_transformers import SentenceTransformer
from streamlit.errors import StreamlitSecretNotFoundError

from config import DATA_DIR, EMBEDDING_MODEL, GEMINI_MODEL, MAX_UPLOAD_BYTES, MIN_RELEVANCE_SCORE, TOP_K
from document_loader import DocumentLoadError, extract_document
from rag_pipeline import (
    GeminiConfigurationError,
    GeminiGenerationError,
    KnowledgeIndex,
    answer_question,
    build_index,
    load_default_knowledge,
    source_fingerprint,
)

st.set_page_config(page_title="Insurance Mitra", page_icon="🛡️", layout="wide")

SUGGESTED_QUESTIONS = [
    "What is the difference between a premium and a deductible?",
    "What documents are commonly needed for a motor insurance claim?",
    "What is a waiting period in health insurance?",
    "How do I identify exclusions in my policy document?",
]


@st.cache_resource(show_spinner="Loading the semantic embedding model...")
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Preparing the shared knowledge base...")
def get_default_knowledge(source_hash: str) -> KnowledgeIndex:
    if not source_hash:
        raise ValueError("No knowledge sources were found.")
    return load_default_knowledge(get_embedder())


def _api_key() -> str | None:
    key = os.getenv("GOOGLE_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("GOOGLE_API_KEY")
    except (FileNotFoundError, AttributeError, StreamlitSecretNotFoundError):
        return None


def _private_knowledge() -> KnowledgeIndex | None:
    cached = st.session_state.get("private_knowledge")
    if cached is not None:
        return cached
    sources = st.session_state.get("private_documents", {})
    if not sources:
        return None
    documents = []
    for name, content in sources.items():
        try:
            documents.extend(extract_document(name, content))
        except DocumentLoadError as exc:
            st.error(str(exc))
            return None
    if not documents:
        return None
    return build_index(documents, get_embedder())


def _render_sources(sources: list) -> None:
    st.markdown("#### RAG Sources")
    if not sources:
        st.caption("No sufficiently relevant knowledge-base passages were found.")
        return
    for number, source in enumerate(sources, start=1):
        page = f" · page {source.page}" if source.page is not None else ""
        with st.expander(f"{number}. {source.source}{page} · relevance {source.score:.2f}"):
            st.write(source.text)


def _ask(question: str, knowledge: KnowledgeIndex, private: KnowledgeIndex | None) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        with st.spinner("Searching the knowledge base and preparing an evidence-based answer..."):
            result = answer_question(
                question,
                knowledge,
                get_embedder(),
                _api_key(),
                model=GEMINI_MODEL,
                top_k=TOP_K,
                min_score=MIN_RELEVANCE_SCORE,
                additional_knowledge=private,
            )
        st.session_state.messages.append(
            {"role": "assistant", "content": result.answer, "sources": result.sources}
        )
    except GeminiConfigurationError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": str(exc), "sources": exc.sources}
        )
    except GeminiGenerationError as exc:
        st.session_state.messages.append(
            {"role": "assistant", "content": str(exc), "sources": exc.sources}
        )
    except APIError:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "Gemini couldn't complete the request. Check your API-key access, configured "
                    "model availability, and network connection, then try again."
                ),
                "sources": [],
            }
        )
    except (RuntimeError, ValueError, OSError) as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"I couldn't complete that request: {exc}",
                "sources": [],
            }
        )


with st.sidebar:
    st.markdown("# 🛡️ Insurance Mitra")
    st.caption("A grounded guide to understanding insurance.")
    st.divider()
    st.subheader("Knowledge base")
    try:
        shared_knowledge = get_default_knowledge(source_fingerprint(DATA_DIR))
        st.success(f"{len(shared_knowledge.chunks)} shared passages indexed")
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Knowledge base unavailable: {exc}")
        st.stop()

    if st.button("Rebuild shared index", use_container_width=True):
        get_default_knowledge.clear()
        try:
            with st.spinner("Rebuilding the index from data/..."):
                shared_knowledge = load_default_knowledge(get_embedder(), force_rebuild=True)
            st.success(f"Rebuilt {len(shared_knowledge.chunks)} passages.")
            st.rerun()
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Index rebuild failed: {exc}")

    st.divider()
    st.subheader("Private policy documents")
    st.caption(
        "PDFs are processed in this app session and are not written to the shared index or "
        "intentionally saved to disk. Relevant excerpts are sent to Google Gemini when you "
        "ask a question. Remove them here or clear the session to discard them. Only upload "
        "documents you are comfortable processing on this server and with Google's API. PDF "
        "parsing is not a malware scan or a guarantee that a file is safe."
    )
    uploads = st.file_uploader(
        "Choose policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="PDF only; maximum 20 MB per file. Scanned PDFs without selectable text may not be searchable.",
        max_upload_size=MAX_UPLOAD_BYTES // (1024 * 1024),
    )
    if st.button("Add to this session", use_container_width=True, disabled=not uploads):
        accepted = dict(st.session_state.get("private_documents", {}))
        errors: list[str] = []
        for upload in uploads or []:
            if upload.size > MAX_UPLOAD_BYTES:
                errors.append(f"{upload.name}: exceeds the 20 MB limit.")
                continue
            content = upload.getvalue()
            try:
                extracted = extract_document(upload.name, content)
                if not extracted:
                    errors.append(f"{upload.name}: no selectable text was found.")
                    continue
                accepted[upload.name] = content
            except DocumentLoadError as exc:
                errors.append(str(exc))
        st.session_state.private_documents = accepted
        st.session_state.pop("private_knowledge", None)
        if accepted:
            try:
                st.session_state.private_knowledge = _private_knowledge()
            except (RuntimeError, ValueError, OSError) as exc:
                st.error(f"Could not index private PDFs in this session: {exc}")
        for error in errors:
            st.warning(error)
        if accepted:
            st.success(f"{len(accepted)} PDF(s) available in this session.")

    private_sources = st.session_state.get("private_documents", {})
    if private_sources:
        for filename in private_sources:
            st.caption(f"• {filename}")
        if st.button("Remove all session PDFs", use_container_width=True):
            st.session_state.private_documents = {}
            st.session_state.pop("private_knowledge", None)
            st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Gemini model: `{GEMINI_MODEL}`")

st.caption("INSURANCE, MADE CLEARER")
st.title("Understand your policy. Make informed decisions.")
st.write(
    "Ask about common insurance concepts and policy language. Insurance Mitra searches its "
    "educational knowledge base first, then answers from relevant retrieved excerpts."
)
st.info(
    "Educational information only. It does not guarantee coverage, claim approval, or legal "
    "or financial outcomes. Your policy wording and applicable regulations determine actual terms."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    st.markdown("### Suggested questions")
    buttons = st.columns(2)
    for index, question in enumerate(SUGGESTED_QUESTIONS):
        with buttons[index % 2]:
            if st.button(question, key=f"suggested-{index}", use_container_width=True):
                _ask(question, shared_knowledge, _private_knowledge())
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            _render_sources(message.get("sources", []))

if prompt := st.chat_input("Ask an insurance question..."):
    _ask(prompt, shared_knowledge, _private_knowledge())
    st.rerun()
