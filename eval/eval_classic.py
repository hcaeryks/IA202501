import joblib
from dataset import get_as_pandas
from preprocessing import preprocess_text
from sklearn.metrics import classification_report, accuracy_score

df = get_as_pandas('test')

X_test = df['text']
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
y_test = df[emotion_cols]

# Preprocess the test data using the same function as training
X_test_processed = preprocess_text(X_test)

pipeline = joblib.load('./model_classic.joblib')

y_pred = pipeline.predict(X_test_processed)

print("\n\n\nClassic model evaluation:\n\n\n ")

accuracy = accuracy_score(y_test, y_pred)
print(f"Exact match accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred, target_names=emotion_cols)
print("Classification report:\n", report)
