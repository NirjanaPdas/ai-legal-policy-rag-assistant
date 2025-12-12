import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def main():

    # 1. Tiny example dataset (you can expand later)

    data = {
        "text": [
            # compliant / safe
            "The company complies with all applicable data protection laws.",
            "Employees must follow all internal security guidelines.",
            "Access to personal data is restricted to authorized staff only.",
            "All financial reports are audited annually by an external auditor.",
            "We encrypt customer data in transit and at rest.",
            "The company maintains clear anti-harassment policies.",
            "Customer data will only be used for purposes described in this policy.",

            # risky / unsafe
            "The company may share user data with third parties without prior notice.",
            "User information can be sold for marketing purposes.",
            "We may access employee emails without any formal process.",
            "Security incidents do not need to be reported to customers.",
            "The company is not responsible for any data breaches.",
            "Personal data may be retained indefinitely without deletion.",
            "There is no clear process for reporting legal or compliance violations.",
        ],
        "label": [
            "compliant",
            "compliant",
            "compliant",
            "compliant",
            "compliant",
            "compliant",
            "compliant",
            "risky",
            "risky",
            "risky",
            "risky",
            "risky",
            "risky",
            "risky",
        ],
    }

    df = pd.DataFrame(data)

    # 2. Train / test split

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    # 3. TF-IDF vectorizer

    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 4. Train classifier

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_vec, y_train)

    # 5. Evaluate

    preds = model.predict(X_test_vec)
    print("=== Classification report ===")
    print(classification_report(y_test, preds))

    # 6. Save model + vectorizer
    # filenames updated to PolicyWise naming
    with open("policywise_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("policywise_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("✅ Training complete.")
    print("Saved: policywise_model.pkl, policywise_vectorizer.pkl")


if __name__ == "__main__":
    main()
