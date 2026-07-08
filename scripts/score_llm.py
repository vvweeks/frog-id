"""
scripts/score_llm.py - Score an LLM's open-ended frog-ID answers against the
ground truth from export_llm_testset.py.

Because the LLM answers open-ended (free-text species names), we map each
answer back to one of the 11 project species using their common names,
scientific names, and known synonyms (from config). Answers that match none
are reported as 'unmatched'; answers that match more than one (e.g. a bare
"leopard frog") are reported as 'ambiguous' for you to adjudicate.

Run:  python -m scripts.score_llm <llm_responses.csv> [--answer-key path]

<llm_responses.csv> is whatever the LLM returned, as two columns:
    clip_id,species
(a header row is fine; extra whitespace and commentary lines are ignored.)
"""
import argparse
import csv
import os
from collections import defaultdict

from config import SPECIES_MAP, SPECIES_SYNONYMS, PROJECT_DIR

EXPORT_DIR = os.path.join(PROJECT_DIR, "llm_testset")
AMBIGUOUS, UNMATCHED = "<ambiguous>", "<unmatched>"


def _build_variants():
    """variant string (lowercased) -> canonical species key."""
    variants = {}
    for common, sci in SPECIES_MAP.items():
        variants[common.replace("_", " ").lower()] = common
        variants[sci.lower()] = common
        for syn in SPECIES_SYNONYMS.get(common, []):
            variants[syn.lower()] = common
    return variants


def match_species(answer, variants):
    """Map a free-text answer to a canonical species key, or AMBIGUOUS/UNMATCHED."""
    a = " ".join(answer.strip().lower().split())
    if not a:
        return UNMATCHED
    if a in variants:                       # exact name/synonym wins
        return variants[a]
    hits = {canon for variant, canon in variants.items() if variant in a or a in variant}
    if len(hits) == 1:
        return next(iter(hits))
    if len(hits) > 1:
        return AMBIGUOUS
    return UNMATCHED


def _load_responses(path):
    """Lenient parse: keep rows whose first cell looks like test_#### and that
    have a second cell. Tolerates headers and stray lines."""
    responses = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 2:
                continue
            clip_id = row[0].strip()
            if not clip_id.lower().startswith("test_"):
                continue
            responses[clip_id] = row[1].strip()
    return responses


def score(responses_path, answer_key_path):
    with open(answer_key_path, newline="") as f:
        truth = {r["clip_id"]: r["true_common_name"].replace(" ", "_")
                 for r in csv.DictReader(f)}
    responses = _load_responses(responses_path)
    variants = _build_variants()

    correct = 0
    unanswered, unmatched, ambiguous = [], [], []
    per_class = defaultdict(lambda: {"n": 0, "correct": 0})
    confusion = defaultdict(int)   # (true, predicted) -> count
    rows_out = []

    for clip_id, true_sp in sorted(truth.items()):
        per_class[true_sp]["n"] += 1
        raw = responses.get(clip_id)
        if raw is None:
            unanswered.append(clip_id)
            pred = "<no answer>"
            is_correct = False
        else:
            matched = match_species(raw, variants)
            pred = matched
            if matched == UNMATCHED:
                unmatched.append((clip_id, raw))
            elif matched == AMBIGUOUS:
                ambiguous.append((clip_id, raw))
            is_correct = (matched == true_sp)
            if is_correct:
                correct += 1
                per_class[true_sp]["correct"] += 1
            if matched not in (UNMATCHED, AMBIGUOUS):
                confusion[(true_sp, matched)] += 1
        rows_out.append({"clip_id": clip_id, "true": true_sp,
                         "llm_raw": raw or "", "matched": pred, "correct": is_correct})

    total = len(truth)
    print(f"\n=== LLM frog-ID score ===")
    print(f"Recordings: {total} | Answered: {total - len(unanswered)} | "
          f"Correct: {correct}  ->  Accuracy {100*correct/total:.1f}% (of all), "
          f"{100*correct/max(1, total-len(unanswered)):.1f}% (of answered)")

    print("\nPer-species accuracy:")
    for sp in sorted(per_class):
        c = per_class[sp]
        print(f"  {sp.replace('_',' '):<24} {c['correct']}/{c['n']}")

    misses = sorted(((t, p, n) for (t, p), n in confusion.items() if t != p),
                    key=lambda x: -x[2])
    if misses:
        print("\nTop confusions (true -> LLM predicted):")
        for t, p, n in misses[:10]:
            print(f"  {t.replace('_',' '):<22} -> {p.replace('_',' '):<22} x{n}")

    if ambiguous:
        print(f"\n⚠️  {len(ambiguous)} ambiguous answer(s) to adjudicate manually:")
        for cid, raw in ambiguous[:10]:
            print(f"    {cid}: \"{raw}\"")
    if unmatched:
        print(f"\n⚠️  {len(unmatched)} answer(s) matched no known species:")
        for cid, raw in unmatched[:10]:
            print(f"    {cid}: \"{raw}\"")
    if unanswered:
        print(f"\n⚠️  {len(unanswered)} recording(s) got no answer from the LLM.")

    out_path = os.path.join(EXPORT_DIR, "scored_results.csv")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "true", "llm_raw", "matched", "correct"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nPer-clip results written to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("responses", help="CSV of the LLM's answers (clip_id,species).")
    parser.add_argument("--answer-key", default=os.path.join(EXPORT_DIR, "answer_key.csv"))
    args = parser.parse_args()
    score(args.responses, args.answer_key)
