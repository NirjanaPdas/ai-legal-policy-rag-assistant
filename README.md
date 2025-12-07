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

### 🎯 Why This Project Is Useful

This project demonstrates:
-How RAG systems work
-How ML and LLMs can work together
-Ability to train and deploy a small classifier
-Practical features used in real companies
-Clear and simple user interface (Streamlit)