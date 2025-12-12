<p align="center">
  <img src="./assets/policywise.png" alt="PolicyWise Screenshot" width="85%">
</p>

<h1 align="center">🛡️ PolicyWise — AI Legal Policy Assistant</h1>

<p align="center">
An intelligent assistant that analyzes policy and legal clauses using RAG, a custom ML risk classifier, and LLM-powered explanations.
</p>

---

## ✨ What is PolicyWise?

PolicyWise is an AI-powered tool that helps Compliance and Legal teams quickly evaluate policy or legal clauses.

It combines:

- **RAG (Retrieval-Augmented Generation)** → Finds relevant text inside uploaded PDF policies  
- **Machine Learning Classifier** → Predicts if a clause is COMPLIANT or RISKY  
- **LLM Explanation (OpenAI)** → Gives clear explanations and safer rewrites  

This makes PolicyWise a smart internal assistant for reviewing documents.

---

## 🚀 Features

### 🔍 1. Document Search (RAG)
Upload PDF policy documents.  
PolicyWise will:

- Extract text  
- Break it into chunks  
- Create embeddings  
- Use FAISS to retrieve the most relevant sections

### 🛡️ 2. Risk Classifier (ML Model)
A Logistic Regression + TF-IDF classifier trained by me.  
It predicts:

- **COMPLIANT**  
- **RISKY**

With a confidence score.

### 🤖 3. AI Explanation (LLM-Enhanced)
If an OpenAI key is provided, PolicyWise can:

- Explain why a clause is risky  
- Highlight dangerous wording  
- Suggest a safer rewrite  
- Use RAG + ML to give better, more contextual answers  

---

## 📁 Project Structure

```bash
policy-wise/
│
├── app.py                 # Main Streamlit application
├── train_model.py         # Training script for ML classifier
├── requirements.txt       # Project dependencies
├── README.md              # Documentation
│
├── policy_model.pkl       # (Optional) Saved ML classifier
├── policy_vectorizer.pkl  # (Optional) Saved TF-IDF vectorizer
│
├── assets/
│   └── policywise.png     # Screenshot for README
│
├── .streamlit/
│   └── config.toml        # Technical blue theme for UI
│
└── .gitignore             # Ignored files (venv, .env, cache, etc.)
```
---

## 🧱 Architecture 
```bash
🧑‍💻 User (Streamlit UI) → 📄 PDF Processing (Extract + Chunk + Embed) → 🔍 FAISS Search (RAG) → 🛡️ ML Classifier (TF-IDF + LR) → 🤖 LLM Review (Explain + Rewrite) → 📤 Final Output
```


## 🛠️ Installation

### 1️⃣ Create a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train the ML model
```bash
python train_model.py
```

### 4️⃣ Run the Streamlit application 
```bash
streamlit run app.py
```

 

