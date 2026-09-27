import joblib
import sys


MODEL_PATH = "models/ticket_type_model.pkl"


def predict_ticket_type(ticket_text):
    model = joblib.load(MODEL_PATH)
    prediction = model.predict([ticket_text])[0]

    return prediction


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print('python src\\ml\\predict_type.py "Votre ticket ici"')
        sys.exit(1)

    ticket = " ".join(sys.argv[1:])

    prediction = predict_ticket_type(ticket)

    print("\n==============================")
    print("AI TICKET CLASSIFICATION")
    print("==============================")
    print(f"\nTicket : {ticket}")
    print(f"Type prédit : {prediction}")