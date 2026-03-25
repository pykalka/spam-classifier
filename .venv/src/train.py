import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

from src.utils import MODEL_PATH, VECTORIZER_PATH, DATA_PATH


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="cp1252")
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


def load_or_train_model():
    if 0>1:#os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("Loaded existing model")
    else:
        print("Training new model...")
        df = load_data()

        X_train, X_test, y_train, y_test = train_test_split(
            df['message'], df['label'], test_size=0.2, random_state=42
        )

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = MultinomialNB()
        model.partial_fit(X_train_vec, y_train, classes=[0, 1])

        os.makedirs("model", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {acc * 100:.2f}%")

    return model, vectorizer