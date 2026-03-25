from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline


def explain_prediction(text, model, vectorizer):
    pipeline = make_pipeline(vectorizer, model)
    explainer = LimeTextExplainer(class_names=['ham', 'spam'])

    exp = explainer.explain_instance(
        text,
        pipeline.predict_proba,
        num_features=10
    )

    exp.save_to_file('lime_explanation.html')
    print("Explanation saved to lime_explanation.html")