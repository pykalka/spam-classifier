import joblib
from src.utils import MODEL_PATH


def online_update(text, true_label, model, vectorizer):
    X_new = vectorizer.transform([text])
    model.partial_fit(X_new, [true_label])
    joblib.dump(model, MODEL_PATH)