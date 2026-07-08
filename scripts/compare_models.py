"""
scripts/compare_models.py - Head-to-head table: for each held-out TEST
recording, show the ground-truth species, your FrogNET model's prediction,
and the LLM's prediction, with overall accuracy for each model.

Runs a trained checkpoint on the current test split (one prediction per
recording), then joins with the LLM answers (parsed + matched to the 11
species by score_llm) and the ground truth from the export's mapping.

Run:
  python -m scripts.compare_models --model resnet  --run-id <run_id> --responses responses.csv
  python -m scripts.compare_models --model birdnet --run-id <run_id> --responses responses.csv
    (or pass --checkpoint <path.pth> instead of --run-id)

The <run_id> is the column in results_log.csv; its checkpoint is
checkpoints/<run_id>.pth. Requires export_llm_testset + score's responses.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import SPECIES_MAP, PROJECT_DIR, CHECKPOINT_DIR, RESULTS_DIR
from scripts.score_llm import _load_responses, match_species, _build_variants, AMBIGUOUS, UNMATCHED

EXPORT_DIR = os.path.join(PROJECT_DIR, "llm_testset")
IDX_TO_SPECIES = list(SPECIES_MAP.keys())


def _predict(model, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
    # dataset is one-sample-per-recording for the test split; align to record_names
    return {name: IDX_TO_SPECIES[p] for name, p in zip(dataset.record_names, preds)}


def resnet_predictions(checkpoint):
    from features.spectrogram_dataset import FrogCallDataset
    from models.resnet_transfer import get_frog_model
    ds = FrogCallDataset("test", SPECIES_MAP)
    model = get_frog_model()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return _predict(model, ds)


def birdnet_predictions(checkpoint, head_type):
    from features.birdnet_embeddings import BirdNETEmbeddingDataset
    from models.birdnet_head import get_birdnet_classifier_head
    train = BirdNETEmbeddingDataset("train", SPECIES_MAP)
    test = BirdNETEmbeddingDataset("test", SPECIES_MAP)
    train_vecs = train.all_train_vectors()  # fit normalization on train (as in training)
    test.set_normalization(train_vecs.mean(axis=0), train_vecs.std(axis=0) + 1e-6)
    model = get_birdnet_classifier_head(embedding_dim=train_vecs.shape[1], head_type=head_type)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    return _predict(model, test)


def _resolve_checkpoint_and_head(args):
    if args.run_id:
        checkpoint = os.path.join(CHECKPOINT_DIR, f"{args.run_id}.pth")
        head_type = args.head
        metrics = os.path.join(RESULTS_DIR, f"{args.run_id}_metrics.json")
        if os.path.exists(metrics):  # prefer the head_type the run was trained with
            head_type = json.load(open(metrics)).get("hyperparams", {}).get("head_type", head_type)
        return checkpoint, head_type
    return args.checkpoint, args.head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet", "birdnet"], required=True)
    parser.add_argument("--run-id", help="Run id (checkpoints/<run_id>.pth + its metrics).")
    parser.add_argument("--checkpoint", help="Explicit checkpoint path (alternative to --run-id).")
    parser.add_argument("--head", choices=["mlp", "linear"], default="mlp",
                        help="BirdNET head type if not inferrable from --run-id metrics.")
    parser.add_argument("--responses", default="responses.csv", help="The LLM's answers CSV.")
    args = parser.parse_args()

    checkpoint, head_type = _resolve_checkpoint_and_head(args)
    if not checkpoint or not os.path.exists(checkpoint):
        raise SystemExit(f"Checkpoint not found: {checkpoint}\nPass --run-id <id> or --checkpoint <path>.")

    mapping_path = os.path.join(EXPORT_DIR, "mapping.csv")
    if not os.path.exists(mapping_path):
        raise SystemExit("mapping.csv not found - run scripts.export_llm_testset first.")
    if not os.path.exists(args.responses):
        raise SystemExit(f"LLM responses not found: {args.responses}")

    # Model predictions (per test recording, keyed by source filename)
    if args.model == "resnet":
        model_preds = resnet_predictions(checkpoint)
    else:
        model_preds = birdnet_predictions(checkpoint, head_type)

    # LLM answers, matched to the 11 species
    variants = _build_variants()
    responses = _load_responses(args.responses)

    mapping = list(csv.DictReader(open(mapping_path)))  # clip_id, species, original_filename
    rows = []
    model_hits = llm_hits = both = neither = 0
    drift = 0
    for m in mapping:
        clip_id, true_sp, fname = m["clip_id"], m["species"], m["original_filename"]

        model_pred = model_preds.get(fname)
        if model_pred is None:
            drift += 1  # file no longer in the current test split
        model_ok = (model_pred == true_sp)

        raw = responses.get(clip_id, "")
        matched = match_species(raw, variants) if raw else "<no answer>"
        llm_pred = matched if matched not in (AMBIGUOUS, UNMATCHED) else raw
        llm_ok = (matched == true_sp)

        model_hits += model_ok
        llm_hits += llm_ok
        both += (model_ok and llm_ok)
        neither += (not model_ok and not llm_ok)
        rows.append({
            "clip_id": clip_id, "filename": fname,
            "ground_truth": true_sp.replace("_", " "),
            "frognet_pred": (model_pred or "<not in test split>").replace("_", " "),
            "frognet_correct": model_ok,
            "llm_pred": llm_pred.replace("_", " "),
            "llm_correct": llm_ok,
        })

    n = len(rows)
    out_dir = EXPORT_DIR if os.path.isdir(EXPORT_DIR) else "."
    out_path = os.path.join(out_dir, "model_vs_llm.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "filename", "ground_truth",
                                          "frognet_pred", "frognet_correct", "llm_pred", "llm_correct"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n=== FrogNET ({args.model}) vs LLM  -  {n} test recordings ===")
    print(f"  FrogNET accuracy: {model_hits}/{n} = {100*model_hits/n:.1f}%")
    print(f"  LLM accuracy:     {llm_hits}/{n} = {100*llm_hits/n:.1f}%")
    print(f"  Both correct: {both} | Both wrong: {neither} | "
          f"Only FrogNET: {model_hits-both} | Only LLM: {llm_hits-both}")
    if drift:
        print(f"  ⚠️  {drift} exported clip(s) aren't in the current test split "
              f"(split changed since export - re-run export_llm_testset to realign).")

    # Per-species accuracy for each model, side by side
    per = defaultdict(lambda: {"n": 0, "m": 0, "l": 0})
    for row, m_ok, l_ok in zip(rows, (r["frognet_correct"] for r in rows), (r["llm_correct"] for r in rows)):
        sp = row["ground_truth"]
        per[sp]["n"] += 1
        per[sp]["m"] += int(m_ok)
        per[sp]["l"] += int(l_ok)
    print(f"\n{'Species':<24}{'n':>4}{'FrogNET':>9}{'LLM':>7}")
    for sp in sorted(per):
        c = per[sp]
        print(f"{sp:<24}{c['n']:>4}{c['m']:>6}/{c['n']:<2}{c['l']:>4}/{c['n']:<2}")

    print(f"\nFull table written to: {out_path}")


if __name__ == "__main__":
    main()
