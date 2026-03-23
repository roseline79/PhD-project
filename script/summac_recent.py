from summac.model_summac import SummaCConv
from datetime import datetime
import csv

source_path = '/home/pogna/rmr451/ECHO/sourcesecho.txt'
prediction_path = '/home/pogna/rmr451/Work2/biobart_aba_echo_fc_v2/generated_predictions.txt'
csv_output_path = '/home/pogna/rmr451/Work2/biobart_aba_echo_fc_v2_summacrecent.csv'

with open(source_path, 'r', encoding='utf-8') as f:
    sources = [line.strip() for line in f if line.strip()]

with open(prediction_path, 'r', encoding='utf-8') as f:
    predictions = [line.strip() for line in f if line.strip()]

n = min(len(sources), len(predictions))
sources = sources[:n]
predictions = predictions[:n]

model = SummaCConv(
    models=["vitc"],
    bins="percentile",
    granularity="sentence",
    nli_labels="e",
    device="cuda",
    start_file="default",
    agg="mean"
)

results = model.score(sources, predictions)

overall_score = sum(results["scores"]) / len(results["scores"])
print(f"\nEvaluated at: {datetime.now().isoformat()}")
print(f"Overall SummaC Score: {overall_score:.4f}")
print(f"Normalized Score (0–100): {overall_score * 100:.2f}")

with open(csv_output_path, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['record_id', 'source', 'generated_prediction', 'summac_score'])
    for i, (src, pred, score) in enumerate(zip(sources, predictions, results["scores"])):
        writer.writerow([i + 1, src, pred, f'{score:.4f}'])