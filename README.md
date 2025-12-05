# ⚖️ AI Legal & Policy Compliance RAG Assistant

This project is a Retrieval-Augmented Generation (RAG) based assistant that helps teams
check if an AI or data usage scenario aligns with their internal company policies.

## ⭐ Features

- Upload policy documents (PDF / TXT)
- Creates embeddings & builds a knowledge base
- **Scenario Compliance Check**
- **Policy Question Answering**
- Retrieves relevant policy chunks for transparency

## 🧠 Tech Stack
- Python
- Streamlit
- OpenAI API
- NumPy
- PyPDF

## 🚀 How to Run

```bash
# 1. Create virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# 2. Install requirements
pip install -r requirements.txt

# 3. Add API key in .env file
OPENAI_API_KEY=your_key_here

# 4. Run the app
streamlit run app.py
