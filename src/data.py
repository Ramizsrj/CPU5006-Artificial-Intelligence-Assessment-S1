# Loads the UCI car evaluation data and splits it into dev/test.
# AI use: written with help from Claude (Anthropic), see README.

import csv
import random
from collections import defaultdict
from pathlib import Path

ATTRIBUTES = ["buying", "maint", "doors", "persons", "lug_boot", "safety"]
CLASS = "class"
CLASSES = ["unacc", "acc", "good", "vgood"]   # ordered worst -> best

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "car.data"


def load_cases(path=DATA_PATH):
    cases = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row:
                continue
            case = dict(zip(ATTRIBUTES + [CLASS], (v.strip() for v in row)))
            cases.append(case)
    return cases


def stratified_split(cases, dev_fraction=0.5, seed=42):
    # dev set = used to look at errors and fix rules
    # test set = only used at the end. stratified so class % stays the same
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for c in cases:
        by_class[c[CLASS]].append(c)
    dev, test = [], []
    for label in CLASSES:
        group = by_class[label][:]
        rng.shuffle(group)
        cut = round(len(group) * dev_fraction)
        dev.extend(group[:cut])
        test.extend(group[cut:])
    return dev, test


def features(case):
    return {a: case[a] for a in ATTRIBUTES}
