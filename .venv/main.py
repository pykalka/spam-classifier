from src.train import load_or_train_model
from src.predict import predict_message
from src.explain import explain_prediction
from src.update import online_update


def main():
    model, vectorizer = load_or_train_model()

    msg = input("\nEnter message: ")

    label, proba = predict_message(msg, model, vectorizer)
    print(f"\nPrediction: {label} ({proba * 100:.2f}%)")

    explain_prediction(msg, model, vectorizer)

    update = input("Update model? (y/n): ").lower()
    if update == 'y':
        true_label = int(input("Enter true label (1 = Spam, 0 = Ham): "))
        online_update(msg, true_label, model, vectorizer)


if __name__ == "__main__":
    main()