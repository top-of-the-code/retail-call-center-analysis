# -------------------- Imports --------------------
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from torch.optim import AdamW
from torch import nn
from tqdm import tqdm
import pickle  # Moved up because encoders are saved early

# -------------------- Utilities & Classes --------------------

def flatten_transcript_segments(segments):
    return " ".join([f"[{s['timestamp']}] {s['speaker'].capitalize()}: {s['text']}" for s in segments])

# ✅ Tokenizer and Label Encoders
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
intent_encoder = LabelEncoder()
subintent_encoder = LabelEncoder()

class TranscriptDataset(Dataset):
    def __init__(self, texts, intents, subintents):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.intent_labels = intents
        self.subintent_labels = subintents

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["intent_labels"] = torch.tensor(self.intent_labels[idx])
        item["subintent_labels"] = torch.tensor(self.subintent_labels[idx])
        return item

    def __len__(self):
        return len(self.intent_labels)

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

# -------------------- Training Code --------------------

if __name__ == "__main__":
    # ✅ Step 1: Load training data
    with open("/content/drive/MyDrive/retail_model_of_call_center/training_data.json") as f:
        raw_data = json.load(f)

    df = pd.DataFrame([
        {
            "text": flatten_transcript_segments(d["Transcript_Segments"]),
            "intent": d["Intent"],
            "sub_intent": d["Sub_Intent"]
        }
        for d in raw_data
        if "Transcript_Segments" in d and "Intent" in d and "Sub_Intent" in d
    ])

    # ✅ Step 2: Encode labels and save encoders
    df["intent_label"] = intent_encoder.fit_transform(df["intent"])
    df["subintent_label"] = subintent_encoder.fit_transform(df["sub_intent"])

    with open("/content/drive/MyDrive/retail_model_of_call_center/intent_encoder.pkl", "wb") as f:
        pickle.dump(intent_encoder, f)
    with open("/content/drive/MyDrive/retail_model_of_call_center/subintent_encoder.pkl", "wb") as f:
        pickle.dump(subintent_encoder, f)
    print("✅ Encoders saved to Google Drive.")

    # ✅ Step 3: Prepare dataset and dataloader using all data
    train_texts = df["text"].tolist()
    train_intents = df["intent_label"].tolist()
    train_subs = df["subintent_label"].tolist()

    train_dataset = TranscriptDataset(train_texts, train_intents, train_subs)
    train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)

    # ✅ Step 4: Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultiOutputBERT(
        num_intents=len(intent_encoder.classes_),
        num_subintents=len(subintent_encoder.classes_)
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=3e-5)
    loss_fn = nn.CrossEntropyLoss()

    # ✅ Step 5: Train model
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

        print(f"✅ Epoch {epoch+1} Loss: {total_loss:.4f}")

    # ✅ Step 6: Save trained model
    torch.save(model.state_dict(), "/content/drive/MyDrive/retail_model_of_call_center/bert_model_for_all_intents.pt")
    print("✅ Model saved to Google Drive.")
