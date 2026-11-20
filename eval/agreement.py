"""Judge-vs-human agreement on a hand-labelled sample (accuracy + Cohen's kappa).

    python -m eval.agreement eval/results/human_label_sample.csv

An LLM judge is only useful if it agrees with people. Rule of thumb for kappa:
>0.8 almost perfect, 0.6-0.8 substantial, 0.4-0.6 moderate.
Labels are collapsed to binary (supported vs not) as well, since the
supported/partial boundary is where humans disagree with each other most.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter


def cohen_kappa(a: list[str], b: list[str]) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main(path: str) -> None:
    with open(path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["human_label"].strip()]
    if not rows:
        print("no human labels filled in yet")
        return
    human = [r["human_label"].strip().lower() for r in rows]
    judge = [r["judge_label"].strip().lower() for r in rows]
    acc = sum(h == j for h, j in zip(human, judge, strict=True)) / len(rows)
    print(f"n={len(rows)}  3-way accuracy={acc:.1%}  kappa={cohen_kappa(human, judge):.3f}")
    hb = ["supported" if h == "supported" else "not" for h in human]
    jb = ["supported" if j == "supported" else "not" for j in judge]
    accb = sum(h == j for h, j in zip(hb, jb, strict=True)) / len(rows)
    print(f"binary accuracy={accb:.1%}  kappa={cohen_kappa(hb, jb):.3f}")
    confusion = Counter(zip(human, judge, strict=True))
    print("\nconfusion (human -> judge):")
    for (h, j), c in sorted(confusion.items()):
        print(f"  {h:22s} -> {j:22s} {c}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "eval/results/human_label_sample.csv")
