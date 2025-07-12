from dataset import get_as_pandas
from preprocessing import preprocess_text
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import AutoConfig
import torch
import torch.nn as nn
from datasets import Dataset
import numpy as np

df = get_as_pandas('train')
X = preprocess_text(df['text']) 
emotion_cols = ['anger', 'fear', 'joy', 'sadness', 'surprise']
y = df[emotion_cols].values.tolist() 
class WeightedDistilBert(DistilBertForSequenceClassification):
    def __init__(self, config, class_weights):
        super().__init__(config)
        self.class_weights = torch.tensor(class_weights).to(self.device)

    def compute_loss(self, model_output, labels):
        logits = model_output.logits
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self.class_weights)
        return loss_fn(logits, labels)

config = AutoConfig.from_pretrained("distilbert-base-uncased", num_labels=5, problem_type='multi_label_classification')
model = WeightedDistilBert.from_pretrained("distilbert-base-uncased", config=config, class_weights=[0.116, 0.558, 0.242, 0.318, 0.289])

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

encodings = tokenizer(X, truncation=True, padding=True, max_length=430)

labels = torch.tensor(y, dtype=torch.float)

dataset = Dataset.from_dict({
    'input_ids': encodings['input_ids'],
    'attention_mask': encodings['attention_mask'],
    'labels': labels
})

training_args = TrainingArguments(
    output_dir='./results_modern',
    per_device_train_batch_size=2,  # Reduced from 8 to 2
    gradient_accumulation_steps=4,  # Maintain effective batch size of 8
    num_train_epochs=2,
    eval_strategy="no",
    save_strategy="epoch",
    logging_dir='./logs',
    save_total_limit=1,
    dataloader_num_workers=0,  # Reduce CPU memory usage
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained('./model_modern')
tokenizer.save_pretrained('./model_modern')
