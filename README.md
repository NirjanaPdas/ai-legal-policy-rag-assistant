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

# AI Legal Policy Assistant (RAG + Risk Classifier)

This project is an AI assistant that helps analyze company policy and legal documents.  
It combines three things:

1. **Document Search (RAG)** → Finds the most relevant parts of uploaded policy PDFs  
2. **Risk Classifier (ML model)** → Detects if a sentence is COMPLIANT or RISKY  
3. **AI Explanation (OpenAI)** → Gives simple explanations and safer rewrites  

It works like a small internal tool used by HR, Legal, and Compliance teams.

---

## ✨ Features

### 🔍 Search inside policy documents
- Upload policy PDF files  
- The system reads and breaks them into small chunks  
- It searches the document to find the most relevant text for your question  

### 🛡️ Compliance Risk Detection (ML Model)
A simple ML model trained by me using Logistic Regression + TF-IDF.  
It predicts:

- **COMPLIANT**  
- **RISKY**  

It also gives a confidence score.

### 🤖 AI Explanation (LLM)
Uses OpenAI to:
- Explain why something is risky  
- Suggest a safer rewrite  
- Give a combined answer using RAG + ML result  
- Use simple, HR-friendly language  


## 📁 Project Structure

app.py                # Main Streamlit application
train_model.py        # Training script for the ML  
                       classifier
policy_model.pkl      # Saved classifier
policy_vectorizer.pkl # Saved TF-IDF vectorizer
requirements.txt
README.md



This project demonstrates:
-How RAG systems work
-How ML and LLMs can work together
-Ability to train and deploy a small classifier
-Practical features used in real companies
-Clear and simple user interface (Streamlit)
