def predict_message(message, model, vectorizer):
    vec = vectorizer.transform([message])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0][pred]

    label = "Spam" if pred else "Ham"
    return label, proba