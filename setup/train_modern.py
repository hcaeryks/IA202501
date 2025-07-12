from dataset import get_as_pandas
from preprocessing import preprocess_text
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from datasets import Dataset
import numpy as np

df = get_as_pandas('train')
X = preprocess_text(df['text']) 
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
y = df[emotion_cols].values.tolist() 

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

encodings = tokenizer(X, truncation=True, padding=True, max_length=128)

labels = torch.tensor(y)

dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels
})

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=5, problem_type='multi_label_classification')

training_args = TrainingArguments(
    output_dir='./results_modern',
    per_device_train_batch_size=8,
    num_train_epochs=2,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained('./model_modern')
tokenizer.save_pretrained('./model_modern')
