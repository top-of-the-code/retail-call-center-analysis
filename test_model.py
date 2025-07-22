from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import BertTokenizerFast
from train_model import MultiOutputBERT, flatten_transcript_segments, intent_encoder, subintent_encoder, TranscriptDataset
import json
import torch
import pandas as pd

# Load test data
with open("test_model.json") as f:
    data = json.load(f)

df = pd.DataFrame([{
    "text": flatten_transcript_segments(d["Transcript_Segments"]),
    "intent": d["Intent"],
    "sub_intent": d["Sub_Intent"]
} for d in data])

df["intent_label"] = intent_encoder.transform(df["intent"])
df["subintent_label"] = subintent_encoder.transform(df["sub_intent"])

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

test_dataset = TranscriptDataset(df["text"].tolist(), df["intent_label"].tolist(), df["subintent_label"].tolist())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MultiOutputBERT(len(intent_encoder.classes_), len(subintent_encoder.classes_)).to(device)
model.load_state_dict(torch.load("bert_multihead.pt", map_location=device))
model.eval()

all_preds_intent, all_preds_sub = [], []
all_true_intent, all_true_sub = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        intent_labels = batch["intent_labels"].to(device)
        subintent_labels = batch["subintent_labels"].to(device)

        intent_logits, subintent_logits = model(input_ids, attention_mask)
        intent_preds = torch.argmax(intent_logits, dim=1)
        subintent_preds = torch.argmax(subintent_logits, dim=1)

        all_preds_intent.extend(intent_preds.cpu().numpy())
        all_preds_sub.extend(subintent_preds.cpu().numpy())
        all_true_intent.extend(intent_labels.cpu().numpy())
        all_true_sub.extend(subintent_labels.cpu().numpy())

print("📊 Intent Classification Report:")
print(classification_report(all_true_intent, all_preds_intent, target_names=intent_encoder.classes_))

print("📊 Sub-Intent Classification Report:")
print(classification_report(all_true_sub, all_preds_sub, target_names=subintent_encoder.classes_))

print(f"✅ Accuracy (Intent): {accuracy_score(all_true_intent, all_preds_intent):.4f}")
print(f"✅ Accuracy (Sub-Intent): {accuracy_score(all_true_sub, all_preds_sub):.4f}")
print(f"✅ F1 Score (Intent): {f1_score(all_true_intent, all_preds_intent, average='weighted'):.4f}")
print(f"✅ F1 Score (Sub-Intent): {f1_score(all_true_sub, all_preds_sub, average='weighted'):.4f}")
