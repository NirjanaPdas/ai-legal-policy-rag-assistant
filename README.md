AI Legal Policy RAG Assistant (with ML Compliance Classifier)

A smart assistant that analyzes legal/policy documents using RAG + OpenAI and classifies text as Compliant or Risky using a custom ML model trained by me.

🚀 Features
🔍 1. RAG-Based Legal Policy Search

Upload a PDF policy document

Text is extracted using pypdf

Vectorized using FAISS

Queries are answered using Retrieval-Augmented Generation (RAG)

Ensures accurate, document-grounded responses

🧠 2. ML Compliance Classifier (Custom Model Trained By Me)

Built a small machine-learning model to classify text as:

COMPLIANT

RISKY

Pipeline built using:

scikit-learn

TfidfVectorizer

LogisticRegression

Trained using train_model.py

Saved as:

policy_model.pkl

policy_vectorizer.pkl

🔎 The app uses this model during inference to highlight potentially risky statements automatically.

🤖 3. AI-Powered Explanatory Answers

Uses OpenAI API to generate clear legal insights

Responses combine:

Retrieved policy sections

Compliance classification

OpenAI explanation

📄 4. Simple Streamlit Frontend

Clean UI to upload file, ask questions, and view classifier output

Real-time predictions and explanations

🛠️ Tech Stack
Layer	Technology
Backend	Python, Scikit-Learn, OpenAI API
Retrieval	FAISS (vector search), custom embeddings
ML Model	TF-IDF + Logistic Regression
Frontend	Streamlit
Storage	Pickle model artifacts
PDF Parsing	pypdf
🧩 Project Structure
ai-legal-policy-rag-assistant/
│── app.py                 # Main Streamlit app
│── train_model.py         # ML model training script
│── policy_model.pkl       # Saved classifier
│── policy_vectorizer.pkl  # Saved TFIDF vectorizer
│── requirements.txt
│── .gitignore
│── README.md

⚙️ How It Works
✔ Step 1 — Train the ML Model

Run once to generate model files:

python train_model.py


Outputs:

policy_model.pkl

policy_vectorizer.pkl

✔ Step 2 — Run the App
streamlit run app.py

🧪 Classifier Example Output

Input:

“The organization may share customer data with external vendors without prior review.”

Output:

Prediction: RISKY  
Confidence: 0.89  
Explanation: “This statement allows uncontrolled sharing of sensitive data.”

⭐ Why This Project Is Strong for Interviews

💡 Shows ability to build real-world RAG systems
💡 Demonstrates ML model training end-to-end
💡 Integrates OpenAI and classical ML
💡 Professional project structure (models, vector DB, app, README)
💡 Solves a real business problem: compliance risk detection

Perfect for AI/ML Engineer, Gen-AI Engineer, SDE (AI focus) roles.

📬 Contact

If you improve or extend the dataset/model, update the .pkl files and re-run the app.