⚖️ **AI Legal Policy RAG Assistant + Compliance Risk Classifier**

A production-style AI system that combines Retrieval-Augmented Generation (RAG), OpenAI LLMs, and a custom-trained ML compliance classifier to analyze legal & organizational policy documents for accurate answers, compliance insights, and risk detection.

This project simulates how modern enterprises build internal AI assistants for policy governance, risk mitigation, and regulatory compliance.

🌐**1. System Overview**
This assistant enables organizations to:

🔍 1.1 Query Any Policy Document Using RAG

-Upload PDF policies
-Extract text using pypdf
-Chunk + embed text
-Store embeddings in FAISS vector database
-Retrieve the most relevant sections
-Generate legally aligned answers with OpenAI

🛡️ 1.2 Automatically Detect Risky Statements (Custom ML Model)
A full ML pipeline that classifies text into:
-COMPLIANT (safe, aligned with policy)
-RISKY (potential legal issues, violations, or      harmful commitments)

The classifier uses:
-TF-IDF Vectorizer
-Logistic Regression
-Trained manually using curated example statements
-Exported as policy_model.pkl and policy_vectorizer.pkl

🤖 1.3 Combined RAG + ML + LLM Workflow
The assistant blends traditional ML + RAG + LLM:
User Query 
   → Retrieve relevant policy sections (FAISS)
   → ML model evaluates risk in the retrieved text
   → OpenAI generates a structured, human-readable response

This hybrid design reflects real enterprise AI architectures used in:
-FinTech
-Insurance
-HR compliance
-Legal-tech
-Governance and risk management

🧠 **2. ML Compliance Classifier Details**
 Algorithm Used:
 -TfidfVectorizer: Converts text → numeric features
 -LogisticRegression: Interpretable, robust baseline classifier

 Training Script:
 train_model.py generates:

 policy_model.pkl          # trained classifier
 policy_vectorizer.pkl     # TF-IDF vectorizer

🎯 **3. Features That Make This Project Enterprise-Ready**
🔐 4.1 No secrets stored in repository
All API keys handled using .env.

⚡ 4.2 Modular Architecture
-train_model.py → ML training
-app.py → Application serving
-Vector DB and LLM calls separated

🛠️ 4.3 Production Practices Included
-.gitignore with sensitive files
-Saved model artifacts for deployment
-Clear documentation & reproducibility
-Streamlit UI for business users

📈 4.4 Extensible Design
You can easily upgrade to:
-Legal entity recognition (NER)
-Multi-label risk classification
-Larger datasets
-Ensembling ML + LLM outputs

🧪 **4. Example Use Cases**
🏢 Corporate Policy Governance:
 HR and Legal teams can validate whether company policies are compliant with regulatory standards.

🔐 Risk & Compliance Automation:
 Automated risk detection for internal audits and employee training.

🔍 Policy Search Engine:
 Quickly retrieve any rule, clause, or requirement across long documents.

📑 Legal Document Assistant:
 Accelerate understanding of service agreements, privacy policies, SOPs, etc.