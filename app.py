import os
import io
from typing import List, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
from pypdf import PdfReader

# Basic config
load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# safety limits so we don't blow up memory
MAX_CHARS = 200_000   # max characters per upload batch
MAX_WORDS = 20_000    # max words for chunking


def get_client(api_key: str | None = None) -> OpenAI:
    """Create OpenAI client from .env or sidebar key."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise ValueError("OpenAI API key not found. Add it in .env or in the sidebar.")
    return OpenAI(api_key=key)

# File → text → chunks
def extract_text_from_pdf(file) -> str:
    """Very simple PDF text extractor."""
    bytes_data = file.read()
    reader = PdfReader(io.BytesIO(bytes_data))

    pages = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt)

    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> List[str]:
    """
    Split long text into overlapping chunks of words.

    Fixed so it cannot get stuck in an infinite loop
    (which was causing MemoryError earlier).
    """
    words = text.split()

    # hard limit to avoid huge uploads eating RAM
    if len(words) > MAX_WORDS:
        words = words[:MAX_WORDS]

    chunks: List[str] = []
    n = len(words)
    start = 0

    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0

    return chunks

# Embeddings + retrieval
def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Create embeddings for a list of texts."""
    all_embs: List[List[float]] = []
    batch_size = 64

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embs = [d.embedding for d in resp.data]
        all_embs.extend(batch_embs)

    return np.array(all_embs, dtype="float32")


def get_query_embedding(client: OpenAI, query: str) -> np.ndarray:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[query],
    )
    return np.array(resp.data[0].embedding, dtype="float32").reshape(1, -1)


def cosine_search(
    doc_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    k: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple cosine similarity search using NumPy only
    (no faiss needed).
    """
    # normalise
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-10
    q_norm = np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10

    doc_normed = doc_embeddings / doc_norms
    q_normed = query_embedding / q_norm

    scores = (doc_normed @ q_normed.T).ravel()          # shape (N,)
    topk_idx = np.argsort(-scores)[:k]                  # highest first
    topk_scores = scores[topk_idx]
    return topk_scores, topk_idx


def build_context(chunks: List[str], indices: List[int]) -> str:
    """Format retrieved chunks into a context string."""
    parts = []
    for rank, idx in enumerate(indices, start=1):
        if 0 <= idx < len(chunks):
            parts.append(f"[Policy Chunk {rank}]\n{chunks[idx]}\n")
    return "\n\n".join(parts)


# LLM logic: analysis + Q&A
def generate_compliance_analysis(
    client: OpenAI,
    scenario: str,
    context: str,
) -> str:
    system_prompt = """
You are an assistant that helps with *policy* and *AI usage* compliance.
You are NOT a lawyer and this is NOT legal advice.

Rules:
- Use ONLY the provided policy context.
- Classify the scenario as one of:
  - Likely Compliant
  - Risky / Needs Review
  - Likely Non-compliant
- Explain the reasoning in simple language.
- Point to the policy chunks you used.
"""

    user_prompt = f"""
Policy Context:
----------------
{context}

Scenario:
---------
{scenario}

Please respond in this structure:

1. Classification: <one of the three options>
2. Reasoning:
   - bullet 1
   - bullet 2
   - bullet 3
3. Referenced chunks: [Policy Chunk X], [Policy Chunk Y]
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def answer_policy_question(
    client: OpenAI,
    question: str,
    context: str,
) -> str:
    system_prompt = """
You answer questions about policies using ONLY the given context.
If the context is not enough, say that clearly.
Do not give real legal advice; this is just an internal helper.
"""

    user_prompt = f"""
Policy Context:
----------------
{context}

Question:
---------
{question}

Give a short, structured answer and mention uncertainty if information is missing.
"""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


# Streamlit UI
def main():
    st.title("⚖️ AI Legal & Policy Compliance RAG Assistant")
    st.write(
        "Upload your company policies and check if AI/data usage scenarios look compliant.\n"
        "_Note: this is a student project, not real legal advice._"
    )

    # ---- Sidebar ----
    with st.sidebar:
        st.subheader("Settings")
        api_key = st.text_input(
            "OpenAI API Key (optional)",
            type="password",
            help="If empty, the app will try OPENAI_API_KEY from your .env file.",
        )

    # ---- File upload ----
    uploaded_files = st.file_uploader(
        "Upload policy documents (PDF or TXT)",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    # store state
    if "policy_chunks" not in st.session_state:
        st.session_state.policy_chunks = None
        st.session_state.policy_embs = None

    if st.button("📦 Build Policy Knowledge Base"):
        if not uploaded_files:
            st.error("Please upload at least one PDF or TXT policy file.")
        else:
            try:
                client = get_client(api_key)
            except ValueError as e:
                st.error(str(e))
                return

            # read + combine text
            full_text = ""
            for f in uploaded_files:
                if f.type == "application/pdf":
                    full_text += extract_text_from_pdf(f) + "\n"
                else:
                    full_text += f.read().decode("utf-8", errors="ignore") + "\n"

            # character safety limit
            if len(full_text) > MAX_CHARS:
                full_text = full_text[:MAX_CHARS]

            chunks = chunk_text(full_text)
            if not chunks:
                st.error("Could not extract any text from the uploaded files.")
                return

            with st.spinner("Creating embeddings..."):
                embs = embed_texts(client, chunks)

            st.session_state.policy_chunks = chunks
            st.session_state.policy_embs = embs
            st.success(f"Knowledge base built with {len(chunks)} chunks ✅")

    st.divider()

    if st.session_state.policy_chunks is None:
        st.info("Upload policies and click **Build Policy Knowledge Base** first.")
        return

    mode = st.radio("Mode", ["Scenario Compliance Check", "Policy Q&A"], horizontal=True)

    # ---- Scenario mode ----
    if mode == "Scenario Compliance Check":
        scenario = st.text_area(
            "Describe your AI / data usage scenario:",
            height=160,
            placeholder="Example: We want to send customer chat logs with phone numbers to a third-party AI API "
                        "for model training.",
        )

        if st.button("🔍 Analyse Compliance"):
            if not scenario.strip():
                st.error("Please type a scenario first.")
                return

            try:
                client = get_client(api_key)
            except ValueError as e:
                st.error(str(e))
                return

            with st.spinner("Retrieving relevant policy chunks..."):
                q_emb = get_query_embedding(client, scenario)
                _, idxs = cosine_search(st.session_state.policy_embs, q_emb, k=5)
                context = build_context(st.session_state.policy_chunks, idxs.tolist())

            with st.spinner("Generating analysis..."):
                analysis = generate_compliance_analysis(client, scenario, context)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Compliance Analysis")
                st.write(analysis)

            with col2:
                st.subheader("Policy Context Used")
                st.write(context)

    # ---- Q&A mode ----
    else:
        question = st.text_area(
            "Ask a question about your policies:",
            height=140,
            placeholder="Example: Are we allowed to share anonymised customer data with an external analytics vendor?",
        )

        if st.button("💬 Answer Question"):
            if not question.strip():
                st.error("Please type a question first.")
                return

            try:
                client = get_client(api_key)
            except ValueError as e:
                st.error(str(e))
                return

            with st.spinner("Retrieving relevant policy chunks..."):
                q_emb = get_query_embedding(client, question)
                _, idxs = cosine_search(st.session_state.policy_embs, q_emb, k=5)
                context = build_context(st.session_state.policy_chunks, idxs.tolist())

            with st.spinner("Generating answer..."):
                answer = answer_policy_question(client, question, context)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Answer")
                st.write(answer)

            with col2:
                st.subheader("Policy Context Used")
                st.write(context)


if __name__ == "__main__":
    main()
