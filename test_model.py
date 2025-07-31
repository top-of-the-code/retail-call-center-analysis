# -------------------- Imports --------------------
import json
import os
import torch
import pickle
import pandas as pd
from torch import nn
import torch.nn.functional as F
from transformers import BertTokenizerFast, BertModel
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
import warnings

# Suppress warnings from sklearn.metrics for classes with no predictions
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option('display.max_rows', 105)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1200)

# -------------------- Configuration Block --------------------
# EASILY RUN IN COLAB: Just modify the paths and parameters here.
class Config:
    MODEL_DIR = "/content/drive/MyDrive/retail_model_of_call_center/"
    TEST_FILE = "/content/drive/MyDrive/retail_model_of_call_center/testing_data.json" # <--- SET YOUR TEST FILE PATH
    OUTPUT_CSV_FILE = "evaluation_results.csv"
    THRESHOLD = 0.08  # Confidence threshold (80%)
    NUM_SAMPLES_TO_DISPLAY = 100

# -------------------- Model and Utility Definitions --------------------
# These must be identical to the definitions used in your training script.

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

def flatten_transcript_segments(segments):
    return " ".join([f"[{s['timestamp']}] {s['speaker'].capitalize()}: {s['text']}" for s in segments])

# -------------------- Core Evaluation Logic --------------------

def run_evaluation(config):
    """Main function to run the entire evaluation process."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Model, Tokenizer, and Encoders ---
    try:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        with open(os.path.join(config.MODEL_DIR, "intent_encoder.pkl"), "rb") as f:
            intent_encoder = pickle.load(f)
        with open(os.path.join(config.MODEL_DIR, "subintent_encoder.pkl"), "rb") as f:
            subintent_encoder = pickle.load(f)

        model = MultiOutputBERT(len(intent_encoder.classes_), len(subintent_encoder.classes_))
        checkpoint = torch.load(os.path.join(config.MODEL_DIR, "bert_model_for_all_intents.pt"), map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print("✅ Model and encoders loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading resources: {e}")
        return

    # --- Load Test Data ---
    try:
        with open(config.TEST_FILE) as f:
            test_data = json.load(f)
        print(f"✅ Loaded {len(test_data)} records from {config.TEST_FILE}")
    except Exception as e:
        print(f"❌ Error loading test file: {e}")
        return

    # --- Process Data and Make Predictions ---
    results_data = []
    for item in tqdm(test_data, desc="Evaluating test data"):
        if "Transcript_Segments" not in item or "Intent" not in item or "Sub_Intent" not in item:
            continue

        text = flatten_transcript_segments(item["Transcript_Segments"])

        # Tokenize
        encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)

        # Predict
        with torch.no_grad():
            intent_logits, subintent_logits = model(input_ids, attention_mask)

        # Get probabilities and raw predictions
        intent_probs = F.softmax(intent_logits, dim=1)
        subintent_probs = F.softmax(subintent_logits, dim=1)

        intent_conf, intent_pred_idx = torch.max(intent_probs, dim=1)
        subintent_conf, subintent_pred_idx = torch.max(subintent_probs, dim=1)

        # Decode raw prediction (for metrics)
        raw_pred_intent = intent_encoder.inverse_transform([intent_pred_idx.item()])[0]
        raw_pred_subintent = subintent_encoder.inverse_transform([subintent_pred_idx.item()])[0]

        # Apply threshold for display/final label
        display_intent = raw_pred_intent if intent_conf.item() >= config.THRESHOLD else "Unclassified"
        display_subintent = raw_pred_subintent if subintent_conf.item() >= config.THRESHOLD else "Unclassified"

        results_data.append({
            'Actual Intent': item["Intent"],
            'Predicted Intent (Raw)': raw_pred_intent,
            'Predicted Intent (Final)': display_intent,
            'Intent Confidence': intent_conf.item(),
            'Actual Sub-Intent': item["Sub_Intent"],
            'Predicted Sub-Intent (Raw)': raw_pred_subintent,
            'Predicted Sub-Intent (Final)': display_subintent,
            'Sub-Intent Confidence': subintent_conf.item(),
            'Transcript': text
        })

    # --- Generate and Display Report ---
    if not results_data:
        print("❌ No valid data found to evaluate.")
        return

    results_df = pd.DataFrame(results_data)
    display_report(results_df, config)

    # --- Save Results ---
    results_df.to_csv(config.OUTPUT_CSV_FILE, index=False)
    print(f"\n✅ Full evaluation results saved to {config.OUTPUT_CSV_FILE}")

def display_report(df, config):
    """Calculates metrics and prints a comprehensive report."""

    # --- Accuracy Scores ---
    print("\n" + "="*80)
    print("ACCURACY REPORT")
    print("="*80)
    # We calculate metrics based on the model's raw (best-guess) predictions
    # This tells us how good the model is, independent of the business threshold rule.
    intent_accuracy = accuracy_score(df['Actual Intent'], df['Predicted Intent (Raw)'])
    subintent_accuracy = accuracy_score(df['Actual Sub-Intent'], df['Predicted Sub-Intent (Raw)'])

    # Overall accuracy: Both raw predictions must be correct
    overall_correct = ((df['Actual Intent'] == df['Predicted Intent (Raw)']) & \
                       (df['Actual Sub-Intent'] == df['Predicted Sub-Intent (Raw)'])).sum()
    overall_accuracy = overall_correct / len(df)

    print(f"Intent Accuracy (Raw Predictions):         {intent_accuracy:.2%}")
    print(f"Sub-Intent Accuracy (Raw Predictions):     {subintent_accuracy:.2%}")
    print(f"Overall Accuracy (Both must be correct): {overall_accuracy:.2%}")

    # --- Classification Reports (Precision, Recall, F1) ---
    print("\n" + "="*80)
    print("PER-CLASS PERFORMANCE (INTENT)")
    print("="*80)
    print(classification_report(df['Actual Intent'], df['Predicted Intent (Raw)'], digits=3))

    print("\n" + "="*80)
    print("PER-CLASS PERFORMANCE (SUB-INTENT)")
    print("="*80)
    print(classification_report(df['Actual Sub-Intent'], df['Predicted Sub-Intent (Raw)'], digits=3))

    # --- Unclassified Stats ---
    unclassified_intent_count = (df['Predicted Intent (Final)'] == 'Unclassified').sum()
    unclassified_subintent_count = (df['Predicted Sub-Intent (Final)'] == 'Unclassified').sum()
    print("\n" + "="*80)
    print(f"THRESHOLD ANALYSIS (Threshold = {config.THRESHOLD:.0%})")
    print("="*80)
    print(f"Intents marked 'Unclassified':     {unclassified_intent_count} / {len(df)} ({unclassified_intent_count/len(df):.2%})")
    print(f"Sub-Intents marked 'Unclassified': {unclassified_subintent_count} / {len(df)} ({unclassified_subintent_count/len(df):.2%})")

    # --- Display Sample Results Table ---
    print("\n" + "="*80)
    print(f"SAMPLE PREDICTIONS (First {min(config.NUM_SAMPLES_TO_DISPLAY, len(df))} entries)")
    print("="*80)

    display_df = df.head(config.NUM_SAMPLES_TO_DISPLAY).copy()
    display_df['Intent Confidence'] = display_df['Intent Confidence'].map('{:.2%}'.format)
    display_df['Sub-Intent Confidence'] = display_df['Sub-Intent Confidence'].map('{:.2%}'.format)
    display_df['Transcript'] = display_df['Transcript'].str.slice(0, 100) + '...'

    print(display_df[[
        'Actual Intent', 'Predicted Intent (Final)', 'Intent Confidence',
        'Actual Sub-Intent', 'Predicted Sub-Intent (Final)', 'Sub-Intent Confidence',
        'Transcript'
    ]].to_string())


# -------------------- Main Execution Block --------------------
if __name__ == "__main__":
    # Initialize config and run the evaluation
    run_evaluation(Config)