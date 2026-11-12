# Hierarchical vs flat rule-based systems - Car Evaluation

Code for my CPU5006 Artificial Intelligence assessment (rule-based AI research paper).

**Research question:** does organising the rules in layers (PRICE, COMFORT, TECH)
like the original DEX model give a better car evaluation expert system than a
flat set of rules that go straight from the inputs to the answer?

## What's in here

```
data/                   car.data and car.names (UCI)
src/engine.py           forward chaining inference engine (used by both systems)
src/kb_hierarchical.py  system A - hierarchical rules
src/kb_flat.py          system B - flat rules
src/data.py             loading the data, dev/test split
src/metrics.py          metrics, McNemar test, bootstrap CI
run_experiment.py       runs everything and makes the figures
results/                output tables, figures and example explanations
```

Both systems use the same engine, only the rules are different. Each rule base has
two versions: v1 is written from domain reasoning only, v2 is after one round of
fixes based on mistakes on the dev set (half the data). The other half (test set)
is only used for the final results. All rules and the reason for every change are
listed in `results/knowledge_bases.txt`.

## How to run

Needs Python 3.10 or newer.

```
pip install -r requirements.txt
python run_experiment.py
```

Takes about a minute. Everything is saved in `results/`.

## Main results (test set, 865 cars)

| System | Accuracy | Macro F1 |
|---|---|---|
| Always guess "unacc" | 0.699 | 0.206 |
| Hierarchical v1 | 0.858 | 0.669 |
| Flat v1 | 0.891 | 0.693 |
| Hierarchical v2 | 0.961 | 0.881 |
| Flat v2 | 0.961 | 0.881 |

- v1: flat is better (McNemar p < 0.001). v2: both give exactly the same answers.
- To add the same fixes the flat rule base went from 9 to 16 rules and 19 to 51
  conditions. The hierarchical one went from 24 to 26 rules and no rule has more
  than 3 conditions.
- Removing one rule at a time: flat can go up to 23.6% silently wrong answers.
  Hierarchical says "unknown" instead of guessing, but when I gave it a default
  rule too (to make it fair) its worst case was 15.5% and its average was a bit
  worse than flat.

## Use of AI

I used Claude (Anthropic) to help write the code and draft the rules, as allowed in
the assessment brief. Each file has a comment saying this.

## Data

Bohanec, M. (1988) *Car Evaluation* [Dataset]. UCI Machine Learning Repository.
Available at: https://doi.org/10.24432/C5JP48
