import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm

# ---------------------- Load and Preprocess ----------------------

def flatten_transcript_segments(segments):
    return " ".join([f"[{s['timestamp']}] {s['speaker'].capitalize()}: {s['text']}" for s in segments])

with open("train_model.json") as f:
    raw_data = json.load(f)

# Flattened format
df = pd.DataFrame([{
    "text": flatten_transcript_segments(d["Transcript_Segments"]),
    "intent": d["Intent"],
    "sub_intent": d["Sub_Intent"]
} for d in raw_data])

# Encode labels
intent_encoder = LabelEncoder()
subintent_encoder = LabelEncoder()
df["intent_label"] = intent_encoder.fit_transform(df["intent"])
df["subintent_label"] = subintent_encoder.fit_transform(df["sub_intent"])

# ---------------------- Tokenization ----------------------

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

class TranscriptDataset(Dataset):
    def __init__(self, texts, intents, subintents):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=684)
        self.intent_labels = intents
        self.subintent_labels = subintents

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["intent_labels"] = torch.tensor(self.intent_labels[idx])
        item["subintent_labels"] = torch.tensor(self.subintent_labels[idx])
        return item

    def __len__(self):
        return len(self.intent_labels)

# Split
train_texts, val_texts, train_intents, val_intents, train_subs, val_subs = train_test_split(
    df["text"], df["intent_label"], df["subintent_label"], test_size=0.2, random_state=42
)

train_dataset = TranscriptDataset(train_texts.tolist(), train_intents.tolist(), train_subs.tolist())
val_dataset = TranscriptDataset(val_texts.tolist(), val_intents.tolist(), val_subs.tolist())

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12)

# ---------------------- Model Definition ----------------------

class MultiOutputBERT(nn.Module):
    def __init__(self, num_intents, num_subintents):
        super(MultiOutputBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.intent_classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.subintent_classifier = nn.Linear(self.bert.config.hidden_size, num_subintents)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.dropout(outputs.pooler_output)
        intent_logits = self.intent_classifier(pooled)
        subintent_logits = self.subintent_classifier(pooled)
        return intent_logits, subintent_logits

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiOutputBERT(
    num_intents=len(intent_encoder.classes_),
    num_subintents=len(subintent_encoder.classes_)
).to(device)

optimizer = AdamW(model.parameters(), lr=3e-5)
loss_fn = nn.CrossEntropyLoss()

# ---------------------- Training ----------------------

for epoch in range(5):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_labels = batch["intent_labels"].to(device)
        subintent_labels = batch["subintent_labels"].to(device)

        intent_logits, subintent_logits = model(input_ids, attention_mask)
        loss = loss_fn(intent_logits, intent_labels) + loss_fn(subintent_logits, subintent_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "bert_multihead.pt")
print("✅ Training Complete. Model Saved.")
