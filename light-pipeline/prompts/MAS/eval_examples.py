#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "evaluate>=0.4.0",
#   "bert-score>=0.3.13",
#   "rouge-score>=0.1.2",
#   "scikit-learn>=1.0",
#   "torch>=2.0",
#   "transformers>=4.30",
#   "sacrebleu>=2.0",
#   "jiwer>=3.0.0",
#   "nltk>=3.8"
# ]
# ///

"""Evaluation Examples using HuggingFace Evaluate Library"""

import evaluate

# Test data
questions = [
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
    "What is 2+2?",
]

predicted = [
    "Paris is the capital of France.",
    "Shakespeare wrote Romeo and Juliet.",
    "2+2 equals 4.",
]

expected = [
    "The capital of France is Paris.",
    "Romeo and Juliet was written by William Shakespeare.",
    "2+2 is 4.",
]

print("\n[DATA] 3 question-answer pairs for evaluation")

# 1. Exact Match
exact_match = evaluate.load("exact_match")
exact_score = exact_match.compute(predictions=predicted, references=expected)
print(f"\n[EXACT MATCH] {exact_score['exact_match']:.3f}")

# 2. BLEU
bleu = evaluate.load("bleu")
bleu_score = bleu.compute(
    predictions=predicted,
    references=[[ans] for ans in expected]
)
print(f"\n[BLEU] {bleu_score['bleu']:.3f}")
print(f"  Precisions: {[f'{p:.3f}' for p in bleu_score['precisions']]}")

# 3. ROUGE
rouge = evaluate.load("rouge")
rouge_scores = rouge.compute(predictions=predicted, references=expected)
print(f"\n[ROUGE]")
print(f"  ROUGE-1: {rouge_scores['rouge1']:.3f}")
print(f"  ROUGE-2: {rouge_scores['rouge2']:.3f}")
print(f"  ROUGE-L: {rouge_scores['rougeL']:.3f}")

# 4. BERTScore
try:
    bertscore = evaluate.load("bertscore")
    bert_scores = bertscore.compute(
        predictions=predicted,
        references=expected,
        lang="en",
        model_type="distilbert-base-uncased",
        device="cpu",
        verbose=False
    )
    avg_f1 = sum(bert_scores['f1']) / len(bert_scores['f1'])
    print(f"\n[BERTSCORE] F1: {avg_f1:.3f}")
except Exception as e:
    print(f"\n[BERTSCORE] Skipped: {str(e)[:50]}...")

# 5. METEOR
meteor = evaluate.load("meteor")
meteor_score = meteor.compute(predictions=predicted, references=expected)
print(f"\n[METEOR] {meteor_score['meteor']:.3f}")

# 6. SacreBLEU
sacrebleu = evaluate.load("sacrebleu")
sacrebleu_score = sacrebleu.compute(
    predictions=predicted,
    references=[[ans] for ans in expected]
)
print(f"\n[SACREBLEU] {sacrebleu_score['score']:.3f}")

# 7. Character Error Rate
try:
    cer = evaluate.load("cer")
    cer_score = cer.compute(predictions=predicted, references=expected)
    print(f"\n[CER] {cer_score:.3f} (lower is better)")
except:
    print("\n[CER] Skipped")

# 8. Word Error Rate
try:
    wer = evaluate.load("wer")
    wer_score = wer.compute(predictions=predicted, references=expected)
    print(f"\n[WER] {wer_score:.3f} (lower is better)")
except:
    print("\n[WER] Skipped")

# Summary
print("\n" + "="*40)
print("Evaluation Summary:")
print(f"  Exact Match: {exact_score['exact_match']:.3f}")
print(f"  BLEU: {bleu_score['bleu']:.3f}")
print(f"  ROUGE-L: {rouge_scores['rougeL']:.3f}")
print(f"  METEOR: {meteor_score['meteor']:.3f}")
print(f"  SacreBLEU: {sacrebleu_score['score']:.3f}")