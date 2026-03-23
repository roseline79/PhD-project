from bert_score import BERTScorer
from datetime import datetime

# 📂 File paths
reference_path = '/home/pogna/rmr451/ECHO/referencesecho.txt'
prediction_path = '/home/pogna/rmr451/Work2/biobart_aba_echo_fc_v2/generated_predictions.txt'

# 📥 Load raw text
with open(reference_path, 'r', encoding='utf-8') as f:
    references = f.readlines()

with open(prediction_path, 'r', encoding='utf-8') as f:
    candidates = f.readlines()

# 🧠 Align lengths
aligned_refs = references[:min(len(references), len(candidates))]
aligned_preds = candidates[:min(len(references), len(candidates))]

# 🧭 Offline scoring with local model
scorer = BERTScorer(
    model_type="/home/pogna/rmr451/roberta-large",  # 👈 Your local model path
    num_layers=17,                      # 👈 Adjust based on model architecture
    lang="en",
    rescale_with_baseline=False,
    idf=False
)

P, R, F1 = scorer.score(aligned_preds, aligned_refs)

# 📊 Print results
print("\n=== Offline BERTScore ===")
print(f"Precision: {P.mean():.4f}")
print(f"Recall:    {R.mean():.4f}")
print(f"F1 Score:  {F1.mean():.4f}")
print(f"🕒 Evaluated at: {datetime.now().isoformat()}")

# 📝 Save individual scores to CSV
import csv
csv_output_path = '/home/pogna/rmr451/biobart_aba_echo_fc_v2_bertscore.csv'
with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['record_id','precision', 'recall', 'f1','reference', 'generated_predictions'])
    for idx, (p, r, f, ref, pred) in enumerate(zip(P.tolist(), R.tolist(), F1.tolist(), aligned_refs, aligned_preds)):
        writer.writerow([
            idx + 1,
            f'{p:.4f}',
            f'{r:.4f}',
            f'{f:.4f}',
            ref.strip(),
            pred.strip()
        ])
