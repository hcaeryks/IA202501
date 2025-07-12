# Memory Optimization Options for Training:
# 1. Use gradient_checkpointing=True to trade compute for memory
# 2. Enable FP16 training with fp16=True for ~50% memory reduction
# 3. Adjust per_device_train_batch_size and gradient_accumulation_steps
# 4. Use dataloader_pin_memory=False to reduce CPU memory usage
# 5. Set max_grad_norm to prevent gradient explosion with less memory

from dataset import get_as_pandas
from preprocessing import preprocess_text
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoConfig
import torch
import torch.nn as nn
from datasets import Dataset
import numpy as np
import gc

# Load and preprocess data
df = get_as_pandas('train')
X = preprocess_text(df['text']) 
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
y = df[emotion_cols].values.tolist()

# Clear dataframe to free memory
del df
gc.collect()

class WeightedDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = torch.tensor(class_weights).to(self.device)

    def compute_loss(self, model_output, labels):
        logits = model_output.logits
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        return loss_fn(logits, labels)

# Load model and tokenizer
config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=5, problem_type='multi_label_classification')
model = WeightedDistilBert.from_pretrained("distilbert-base-uncased", config=config, class_weights=[0.116, 0.558, 0.242, 0.318, 0.289])

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

# Tokenize data
print("Tokenizing data...")
encodings = tokenizer(X, truncation=True, padding=True, max_length=430)

# Convert labels to tensor
labels = torch.tensor(y, dtype=torch.float)

# Clear intermediate variables
del X, y
gc.collect()

# Create dataset
dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels
})

# Clear encodings and labels to free memory
del encodings, labels
gc.collect()

# Memory-optimized training arguments
training_args = TrainingArguments(
    output_dir='./results_modern',
    per_device_train_batch_size=1,  # Reduced from 2 to 1 for maximum memory efficiency
    gradient_accumulation_steps=8,  # Maintain effective batch size of 8
    num_train_epochs=2,
    eval_strategy="no",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=1,
    dataloader_num_workers=0,  # Reduce CPU memory usage
    dataloader_pin_memory=False,  # Reduce CPU memory usage
    gradient_checkpointing=True,  # Trade compute for memory
    fp16=True,  # Enable mixed precision training (~50% memory reduction)
    max_grad_norm=1.0,  # Prevent gradient explosion
    logging_steps=50,  # Reduce logging frequency
    remove_unused_columns=False,  # Keep all columns for compatibility
    report_to="none",  # Disable wandb/tensorboard to save memory
)

print("Starting training with memory-optimized settings...")
print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"FP16 training: {training_args.fp16}")
print(f"Gradient checkpointing: {training_args.gradient_checkpointing}")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

# Train model
trainer.train()

# Save model and tokenizer
print("Saving model...")
model.save_pretrained('./model_modern')
tokenizer.save_pretrained('./model_modern')

print("Training completed successfully!")
