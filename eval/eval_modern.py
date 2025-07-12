from dataset import get_as_pandas
from preprocessing import preprocess_text
from transformers import BertTokenizerFast, BertForSequenceClassification
import torch
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

df = get_as_pandas('test')
X_test = preprocess_text(df['text'])
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
y_test = df[emotion_cols].values

model_path = './model_modern'
tokenizer = BertTokenizerFast.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

inputs = tokenizer(
    X_test,
    padding=True,
    truncation=True,
    max_length=128,
    return_tensors='pt'
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

probs = torch.sigmoid(logits)

threshold = 0.5
y_pred = (probs.numpy() >= threshold).astype(int)

print("\n\nModern model evaluation:\n\n")

accuracy = accuracy_score(y_test, y_pred)
print(f"Exact match accuracy: {accuracy:.4f}\n")

report = classification_report(y_test, y_pred, target_names=emotion_cols)
print("Classification report:\n", report)
