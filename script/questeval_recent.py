#!/usr/bin/env python3
import csv
import json
import sys

try:
    from questeval.questeval_metric import QuestEval
except ModuleNotFoundError as exc:
    if exc.name == "evaluate":
        print("Missing dependency: 'evaluate' is required by QuestEval.")
        print("Install it in your active environment and rerun:")
        print("  pip install evaluate")
        print("or")
        print("  conda install -c conda-forge evaluate")
        sys.exit(1)
    raise

SRC_PATH = "/home/pogna/rmr451/ECHO/sourcesecho.txt"
HYP_PATH = "/home/pogna/rmr451/Work2/biobart_aba_echo_fc_v2/generated_predictions.txt"
CSV_OUT = "/home/pogna/rmr451/Work2/biobart_aba_echo_fc_v2_questeval_recent_scores.csv"
JSON_OUT = "/home/pogna/rmr451/Work2/biobart_aba_echo_fc_v2_questeval_recent_scores.json"

def read_nonempty_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

sources = read_nonempty_lines(SRC_PATH)
hypotheses = read_nonempty_lines(HYP_PATH)

n = min(len(sources), len(hypotheses))
sources = sources[:n]
hypotheses = hypotheses[:n]

questeval = QuestEval(no_cuda=False)

scores = questeval.corpus_questeval(
    hypothesis=hypotheses,
    sources=sources
)

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["index", "source", "prediction", "questeval_score"])
    for i, (src, hyp, s) in enumerate(zip(sources, hypotheses, scores["ex_level_scores"])):
        writer.writerow([i, src, hyp, f"{s:.6f}"])

with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(scores, f, indent=2)

print(f'Corpus QuestEval: {scores["corpus_score"]:.6f}')
print(f"Saved CSV to: {CSV_OUT}")
print(f"Saved JSON to: {JSON_OUT}")