# Memory Optimization Options:
# 1. Reduce batch_size below (e.g., 16 or 8) if still running out of memory
# 2. Use model.half() for FP16 inference to reduce memory by ~50%
# 3. Process data in streaming mode if dataset is very large
# 4. Use torch.jit.script(model) for memory-efficient inference

from dataset import get_as_pandas
from preprocessing import preprocess_text
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import torch
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import gc

# Load and preprocess data
df = get_as_pandas('test')
X_test = preprocess_text(df['text'])
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
y_test = df[emotion_cols].values

# Clear the original dataframe to free memory
del df
gc.collect()

# Load model and tokenizer
model_path = './model_modern'
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
model = DistilBertForSequenceClassification.from_pretrained(model_path)

# Optional: Use half precision for further memory reduction
# model = model.half()  # Uncomment for FP16 inference

model.eval()

# Batch processing to reduce memory usage
batch_size = 32  # Adjust based on your available memory (try 16, 8, or 4 if needed)
threshold = 0.5

all_predictions = []
total_samples = len(X_test)

print(f"Processing {total_samples} samples in batches of {batch_size}...")

for i in range(0, total_samples, batch_size):
    # Get batch
    batch_texts = X_test[i:i+batch_size]
    
    # Tokenize batch
    inputs = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=430,
        return_tensors='pt'
    )
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert to probabilities and predictions
    probs = torch.sigmoid(logits)
    batch_predictions = (probs.numpy() >= threshold).astype(int)
    
    # Store predictions
    all_predictions.extend(batch_predictions)
    
    # Clear batch variables to free memory
    del inputs, outputs, logits, probs, batch_predictions, batch_texts
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Progress indicator
    if (i // batch_size + 1) % 10 == 0:
        print(f"Processed {min(i+batch_size, total_samples)}/{total_samples} samples")

# Convert predictions to numpy array
y_pred = np.array(all_predictions)

# Clear intermediate variables
del all_predictions, X_test
gc.collect()

print("\n\n\nModern model evaluation:\n\n\n ")

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"Exact match accuracy: {accuracy:.4f}")

report = classification_report(y_test, y_pred, target_names=emotion_cols)
print("Classification report:\n", report)
