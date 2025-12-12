import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

import numpy as np
from pypdf import PdfReader
import pickle
import faiss
from pathlib import Path
import os

st.set_page_config(
    page_title="PolicyWise — Smart Policy Guidance",
    page_icon="⚖️",
    layout="wide",
)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client (only if key is present)
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

BASE_DIR = Path(__file__).resolve().parent
# renamed artifact files to include the project name PolicyWise
MODEL_PATH = BASE_DIR / "policywise_model.pkl"
VECTORIZER_PATH = BASE_DIR / "policywise_vectorizer.pkl"

# in-memory objects so we load them only once
_model = None
_vectorizer = None

# ML PART (COMPLIANT / RISKY)

def load_policy_model():
    """Load trained logistic regression model + TF-IDF vectorizer."""
    global _model, _vectorizer

    if _model is not None and _vectorizer is not None:
        return _model, _vectorizer

    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        return None, None

    with open(MODEL_PATH, "rb") as f:
        _model = pickle.load(f)

    with open(VECTORIZER_PATH, "rb") as f:
        _vectorizer = pickle.load(f)

    return _model, _vectorizer


def predict_policy_risk(text: str):
    """
    Predict if a clause is COMPLIANT or RISKY.
    Returns (label, confidence between 0 and 1).
    """
    model, vec = load_policy_model()
    if model is None or vec is None:
        return "model_not_available", 0.0

    X = vec.transform([text])
    proba = model.predict_proba(X)[0]
    label = model.predict(X)[0]

    class_index = list(model.classes_).index(label)
    conf = float(proba[class_index])
    return label, conf

# RAG HELPERS (PDF + FAISS)

def init_session():
    """Create session_state variables if not present."""
    if "faiss_index" not in st.session_state:
        st.session_state.faiss_index = None
    if "policy_chunks" not in st.session_state:
        st.session_state.policy_chunks = []
    if "chunk_embeddings" not in st.session_state:
        st.session_state.chunk_embeddings = None
    if "llm_answer" not in st.session_state:
        st.session_state.llm_answer = ""


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read text from a PDF file (simple version)."""
    reader = PdfReader(io := bytes_to_stream(file_bytes))
    texts = []
    for page in reader.pages:
        try:
            t = page.extract_text() or ""
            texts.append(t.replace("\n", " "))
        except Exception:
            pass
    return "\n".join(texts)


def bytes_to_stream(b: bytes):
    # small helper so we don’t import io everywhere
    import io as _io
    return _io.BytesIO(b)


def split_into_chunks(text: str, chunk_size: int = 600, overlap: int = 120):
    """Split big text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


def create_embeddings(texts):
    """Create OpenAI embeddings for a list of texts."""
    if client is None:
        return None

    try:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        vectors = [d.embedding for d in resp.data]
        return np.array(vectors, dtype="float32")
    except Exception:
        # if no credit / error, we just disable RAG
        return None


def build_faiss_index(chunks):
    """Create FAISS index from text chunks."""
    emb = create_embeddings(chunks)
    if emb is None:
        return None, None

    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    return index, emb


def search_chunks(query: str, k: int = 5):
    """Get top-k relevant chunks from the index."""
    if st.session_state.faiss_index is None:
        return []

    q_emb = create_embeddings([query])
    if q_emb is None:
        return []

    D, I = st.session_state.faiss_index.search(q_emb, k)
    indices = I[0]
    results = []
    for idx in indices:
        if 0 <= idx < len(st.session_state.policy_chunks):
            results.append(st.session_state.policy_chunks[idx])
    return results


# LLM REVIEW (USES RAG CONTEXT)

def llm_available() -> bool:
    return client is not None


def llm_review_clause(clause: str, context_chunks):
    """
    Ask LLM to review the clause using retrieved policy context.
    If no key / credit, return a simple text explanation.
    """
    context = "\n\n---\n\n".join(context_chunks[:5]) if context_chunks else "No policy context available."

    prompt = f"""
You are a legal/compliance assistant for a company.

Clause:
\"\"\"{clause}\"\"\" 

Relevant policy snippets:
{context}

Tasks:
1. Say if the clause looks mostly compliant or risky, and why.
2. Point out any dangerous / vague phrases.
3. Suggest a safer rewrite that would be more compliant.
Use clear simple English.
"""

    if not llm_available():
        return (
            "LLM review is in demo mode (no API key or credits).\n\n"
            "In a real deployment this section would contain a detailed "
            "legal/compliance analysis generated by an LLM using the policy snippets above."
        )

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content
    except Exception as e:
        return (
            "LLM call failed (probably quota or network).\n\n"
            f"Error: {e}\n\n"
            "You can still use the ML classifier and retrieved policy snippets as a quick signal."
        )


# UI
init_session()

st.markdown(
    """
    <h2 style="margin-bottom:0.2rem;">PolicyWise</h2>
    <p style="font-size:0.9rem; color:#bbbbbb;">
    Paste a policy clause, upload your policy PDFs, and get:
    <br>• a simple ML risk score (COMPLIANT / RISKY)
    <br>• relevant policy snippets using RAG
    <br>• an optional LLM explanation.
    </p>
    """,
    unsafe_allow_html=True,
)

# Sidebar: PDF upload and indexing
with st.sidebar:
    st.header("📚 PolicyWise Documents")
    files = st.file_uploader(
        "Upload policy PDFs (data protection, HR, security, etc.)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if st.button("Build / refresh index"):
        full_text = ""
        for f in files or []:
            try:
                pdf_bytes = f.read()
                text = extract_text_from_pdf(pdf_bytes)
                full_text += "\n" + text
            except Exception:
                pass

        if not full_text.strip():
            st.warning("Could not read any text from PDFs.")
        else:
            chunks = split_into_chunks(full_text)
            index, emb = build_faiss_index(chunks)
            if index is None:
                st.error(
                    "Could not create embeddings. Probably no OpenAI key/credits. "
                    "RAG will be disabled but ML classifier still works."
                )
            else:
                st.session_state.faiss_index = index
                st.session_state.policy_chunks = chunks
                st.session_state.chunk_embeddings = emb
                st.success(f"Index built with {len(chunks)} chunks.")

    if st.session_state.faiss_index is not None:
        st.caption(f"Indexed chunks: {len(st.session_state.policy_chunks)} (RAG ready)")
    else:
        st.caption("No index yet. Upload PDFs and click the button above.")


# Main layout
col_left, col_right = st.columns([1.4, 1.0])

# Left: clause + LLM review
with col_left:
    st.subheader("📜 Clause")
    clause_text = st.text_area(
        "Paste a single clause or short section:",
        height=160,
        placeholder=(
            "Example: The company may share customer data with third parties "
            "for marketing without explicit consent."
        ),
    )

    analyze_btn = st.button("Analyze clause")

    st.markdown("---")
    st.subheader("🤖 LLM review (with policy context)")

    if analyze_btn and clause_text.strip():
        with st.spinner("Getting policy context and generating answer..."):
            ctx_chunks = search_chunks(clause_text.strip())
            st.session_state.llm_answer = llm_review_clause(clause_text.strip(), ctx_chunks)
    elif analyze_btn and not clause_text.strip():
        st.warning("Please paste a clause first.")

    if clause_text.strip() and st.session_state.llm_answer:
        st.write(st.session_state.llm_answer)
    else:
        st.info("Paste a clause and click **Analyze clause** to see the review here.")

# Right: ML risk + context snippets
with col_right:
    st.subheader("🔎 ML risk classifier")

    if analyze_btn and clause_text.strip():
        label, conf = predict_policy_risk(clause_text.strip())
        if label == "model_not_available":
            st.error(
                "Trained model not found. Run `python train_model.py` once "
                "to create policywise_model.pkl and policywise_vectorizer.pkl."
            )
        else:
            emoji = "🟢" if label == "compliant" else "🔴"
            st.markdown(
                f"**{emoji} Prediction:** `{label.upper()}`  "
                f"(confidence: {conf * 100:.1f}%)"
            )
            st.caption("Simple logistic regression model trained on example clauses.")
    else:
        st.info("After you click Analyze, the ML prediction will show here.")

    st.markdown("---")
    st.subheader("📎 Retrieved policy snippets (RAG)")

    if analyze_btn and clause_text.strip():
        snippets = search_chunks(clause_text.strip())
        if not snippets:
            st.write("No snippets available (no index or embeddings disabled).")
        else:
            for i, snip in enumerate(snippets[:3], start=1):
                st.markdown(f"**Snippet {i}:**")
                st.write(snip)
    else:
        st.write("After analysis, policy snippets related to the clause will appear here.")

st.markdown(
    "<p style='font-size:0.75rem; color:#888888; margin-top:1rem;'>"
    "Project: PolicyWise – combines a small trained ML model with RAG and optional LLM review."
    "</p>",
    unsafe_allow_html=True,
)
