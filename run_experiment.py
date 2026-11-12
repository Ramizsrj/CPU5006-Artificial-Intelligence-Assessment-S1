# Runs all the experiments and saves tables + figures to results/
#   python run_experiment.py
#
# E1 - accuracy / macro F1 on the test set (+ baseline, bootstrap CI, McNemar)
# E2 - size of each rule base and how inference behaves
# E3 - remove one rule at a time and see what breaks (with a control:
#      hierarchical + default rule, so it is a fair comparison with flat)
# E4 - example explanations
#
# AI use: written with help from Claude (Anthropic), see README.

import csv
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import kb_flat
import kb_hierarchical
from data import CLASSES, features, load_cases, stratified_split
from engine import ForwardChainingEngine, Rule
from metrics import (bootstrap_ci, classification_metrics, confusion,
                     mcnemar_exact)

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

# colours for the charts
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e6e5e0"
SYSTEM_COLOUR = {"Hierarchical": BLUE, "Flat": ORANGE, "Hierarchical + default": AQUA}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "savefig.dpi": 200, "savefig.bbox": "tight",
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SYSTEMS = {"Hierarchical": kb_hierarchical, "Flat": kb_flat}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(x):
    return round(x, 4) if isinstance(x, float) else x


def main():
    cases = load_cases()
    dev, test = stratified_split(cases, dev_fraction=0.5, seed=42)
    y_test = [c["class"] for c in test]
    print(f"Loaded {len(cases)} cases -> development {len(dev)}, test {len(test)}")

    # ---- E1 ----
    preds, perf_rows = {}, []
    baseline = ["unacc"] * len(test)   # always guess the biggest class
    preds[("Baseline", 0)] = baseline
    for name, kb in SYSTEMS.items():
        for v in (1, 2):
            eng = ForwardChainingEngine(kb.build_rules(v), "class")
            preds[(name, v)] = [eng.classify(features(c)) for c in test]

    for (name, v), p in preds.items():
        m = classification_metrics(y_test, p)
        lo, hi = bootstrap_ci(y_test, p, "macro_f1")
        row = {"system": name, "version": v if v else "-", **m,
               "macro_f1_ci_low": lo, "macro_f1_ci_high": hi,
               "unclassified": sum(1 for x in p if x == "unknown")}
        perf_rows.append({k: fmt(val) for k, val in row.items()})
    write_csv(RESULTS / "E1_test_performance.csv", perf_rows)

    mcn_rows = []
    for v in (1, 2):
        b, c, pval = mcnemar_exact(y_test, preds[("Hierarchical", v)], preds[("Flat", v)])
        agree = mean(1.0 if a == f else 0.0
                     for a, f in zip(preds[("Hierarchical", v)], preds[("Flat", v)]))
        mcn_rows.append({"comparison": f"A = Hierarchical v{v}, B = Flat v{v}",
                         "A_right_B_wrong": b, "A_wrong_B_right": c,
                         "p_value": ("<0.0001" if pval < 1e-4 else fmt(pval)), "prediction_agreement": fmt(agree)})
    for name in SYSTEMS:
        b, c, pval = mcnemar_exact(y_test, preds[(name, 2)], preds[(name, 1)])
        mcn_rows.append({"comparison": f"A = {name} v2, B = {name} v1",
                         "A_right_B_wrong": b, "A_wrong_B_right": c,
                         "p_value": ("<0.0001" if pval < 1e-4 else fmt(pval)), "prediction_agreement": ""})
    write_csv(RESULTS / "E1_mcnemar.csv", mcn_rows)

    cms = {k: confusion(y_test, p) for k, p in preds.items() if k[0] != "Baseline"}

    # ---- E2 ----
    cost_rows = []
    for name, kb in SYSTEMS.items():
        for v in (1, 2):
            rules = kb.build_rules(v)
            eng = ForwardChainingEngine(rules, "class")
            runs = [eng.infer(features(c)) for c in cases]   # whole dataset
            n_class_rules = sum(1 for r in rules if r.conclusion[0] == "class")
            cost_rows.append({
                "system": name, "version": v,
                "rules_total": len(rules),
                "rules_concluding_class": n_class_rules,
                "rules_intermediate": len(rules) - n_class_rules,
                "conditions_total": sum(r.specificity for r in rules),
                "conditions_per_rule": fmt(mean(r.specificity for r in rules)),
                "max_conditions_in_a_rule": max(r.specificity for r in rules),
                "rules_changed_or_added_in_v2": (len(kb.REFINEMENTS) if v == 2 else 0),
                "rules_fired_per_case": fmt(mean(r.cycles for r in runs)),
                "cases_with_conflict_pct": fmt(100 * mean(1.0 if r.conflicts else 0.0 for r in runs)),
                "cases_with_contradiction_pct": fmt(100 * mean(1.0 if r.contradictions else 0.0 for r in runs)),
                "cases_decided_by_default_pct": fmt(100 * mean(1.0 if r.used_default else 0.0 for r in runs)),
                "rules_never_fired": sum(1 for rule in rules
                                         if not any(rule in r.trace for r in runs)),
            })
    write_csv(RESULTS / "E2_knowledge_cost.csv", cost_rows)

    # ---- E3 ----
    # control: hierarchical + the same default rule as flat, otherwise the
    # comparison isn't fair (flat has a default, hierarchical doesn't)
    ablation_bases = {name: kb.build_rules(2) for name, kb in SYSTEMS.items()}
    ablation_bases["Hierarchical + default"] = kb_hierarchical.build_rules(2) + [
        Rule("H0", {}, ("class", "unacc"), -1, "default: reject if nothing else applies")]
    ablation_rows = []
    for name, full in ablation_bases.items():
        for removed in full:
            if removed.name == "H0":
                continue
            eng = ForwardChainingEngine([r for r in full if r is not removed], "class")
            p = [eng.classify(features(c)) for c in test]
            m = classification_metrics(y_test, p)
            ablation_rows.append({
                "system": name, "removed_rule": removed.name,
                "accuracy": fmt(m["accuracy"]), "macro_f1": fmt(m["macro_f1"]),
                "unclassified_pct": fmt(100 * mean(1.0 if x == "unknown" else 0.0 for x in p)),
                "silently_wrong_pct": fmt(100 * mean(1.0 if (x != "unknown" and x != t) else 0.0
                                                     for x, t in zip(p, y_test))),
            })
    write_csv(RESULTS / "E3_rule_ablation.csv", ablation_rows)

    # ---- E4 ----
    examples = [
        {"buying": "low", "maint": "med", "doors": "4", "persons": "more",
         "lug_boot": "big", "safety": "high"},
        {"buying": "high", "maint": "med", "doors": "2", "persons": "4",
         "lug_boot": "small", "safety": "med"},
    ]
    with open(RESULTS / "E4_explanations.txt", "w", encoding="utf-8") as f:
        for name, kb in SYSTEMS.items():
            eng = ForwardChainingEngine(kb.build_rules(2), "class")
            f.write(f"===== {name} system (version 2) =====\n")
            for ex in examples:
                f.write(eng.explain(ex) + "\n\n")
    with open(RESULTS / "knowledge_bases.txt", "w", encoding="utf-8") as f:
        for name, kb in SYSTEMS.items():
            f.write(f"===== {name} knowledge base (version 2) =====\n")
            for r in kb.build_rules(2):
                f.write(f"[priority {r.priority}] {r}\n")
            f.write("\nRefinements made from development-set errors:\n")
            for target, rule, reason in kb.REFINEMENTS:
                action = f"replaced {target}" if target else "added"
                f.write(f"  - {rule.name} ({action}): {reason}\n")
            f.write("\n")

    # ---- figures ----
    fig_performance(perf_rows)
    fig_recall(perf_rows)
    fig_confusion(cms)
    fig_cost(cost_rows)
    fig_ablation(ablation_rows, perf_rows)

    # ---- print summary ----
    print("\nE1 - held-out test set")
    for r in perf_rows:
        print(f"  {r['system']:<12} v{r['version']}  acc={r['accuracy']:.3f}  "
              f"macroF1={r['macro_f1']:.3f}  [{r['macro_f1_ci_low']:.3f}, {r['macro_f1_ci_high']:.3f}]")
    for r in mcn_rows:
        print(f"  McNemar {r['comparison']}: b={r["A_right_B_wrong"]} "
              f"c={r["A_wrong_B_right"]} p={r['p_value']}")
    print("\nE2 - knowledge cost")
    for r in cost_rows:
        print("  ", r)
    print("\nE3 - rule ablation (mean / worst)")
    for name in ["Hierarchical", "Hierarchical + default", "Flat"]:
        x = [r for r in ablation_rows if r["system"] == name]
        print(f"  {name:<24} macroF1 {mean(r['macro_f1'] for r in x):.3f} / "
              f"{min(r['macro_f1'] for r in x):.3f}   silent wrong % "
              f"{mean(r['silently_wrong_pct'] for r in x):.2f} / "
              f"{max(r['silently_wrong_pct'] for r in x):.2f}   unclassified % "
              f"{mean(r['unclassified_pct'] for r in x):.2f}")
    print(f"\nAll outputs written to {RESULTS}")


# ---- chart functions ----
def _get(rows, system, version):
    return next(r for r in rows if r["system"] == system and str(r["version"]) == str(version))


def fig_performance(rows):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(2)
    w = 0.34
    for i, name in enumerate(SYSTEMS):
        vals = [_get(rows, name, v)["macro_f1"] for v in (1, 2)]
        lo = [vals[j] - _get(rows, name, v)["macro_f1_ci_low"] for j, v in enumerate((1, 2))]
        hi = [_get(rows, name, v)["macro_f1_ci_high"] - vals[j] for j, v in enumerate((1, 2))]
        pos = x + (i - 0.5) * (w + 0.02)
        ax.bar(pos, vals, w, color=SYSTEM_COLOUR[name], label=name, zorder=2)
        ax.errorbar(pos, vals, yerr=[lo, hi], fmt="none", ecolor=INK2, capsize=3, lw=1, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, 0.03, f"{v:.3f}", ha="center", va="bottom", color="white",
                    fontsize=9, fontweight="bold")
    base = _get(rows, "Baseline", "-")["macro_f1"]
    ax.axhline(base, color=MUTED, ls="--", lw=1.2, zorder=1,
               label=f"Majority-class baseline ({base:.3f})")
    ax.set_xticks(x, ["Version 1\n(domain knowledge only)", "Version 2\n(after one refinement)"])
    ax.set_ylabel("Macro-F1 on held-out test set")
    ax.set_ylim(0, 1.05)
    ax.set_ylim(0, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8.5)
    ax.set_title("Classification performance (95% bootstrap CI)", loc="left", fontsize=10.5)
    fig.savefig(RESULTS / "fig1_test_macro_f1.png")
    plt.close(fig)


def fig_recall(rows):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    x = np.arange(len(CLASSES))
    w = 0.38
    for ax, v in zip(axes, (1, 2)):
        for i, name in enumerate(SYSTEMS):
            vals = [_get(rows, name, v)[f"recall_{c}"] for c in CLASSES]
            ax.bar(x + (i - 0.5) * (w + 0.02), vals, w, color=SYSTEM_COLOUR[name],
                   label=name, zorder=2)
        ax.set_xticks(x, CLASSES)
        ax.set_title(f"Version {v}", loc="left", fontsize=10)
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Recall (test set)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Per-class recall: rare classes are where the systems differ",
                 x=0.01, y=1.0, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(RESULTS / "fig2_per_class_recall.png")
    plt.close(fig)


def fig_confusion(cms):
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 7.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "blues", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95"])
    for ax, key in zip(axes.flat, [("Hierarchical", 1), ("Flat", 1),
                                   ("Hierarchical", 2), ("Flat", 2)]):
        cm = cms[key]
        norm = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                        color="white" if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(4), CLASSES)
        ax.set_yticks(range(4), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        ax.set_title(f"{key[0]}, version {key[1]}", loc="left", fontsize=10)
    fig.suptitle("Confusion matrices on the test set (counts; shading = row %)",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(RESULTS / "fig3_confusion_matrices.png")
    plt.close(fig)


def fig_cost(rows):
    metrics = [("rules_total", "Rules"), ("conditions_total", "Conditions (total)"),
               ("max_conditions_in_a_rule", "Most conditions in one rule")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2))
    x = np.arange(2)
    w = 0.38
    for ax, (key, label) in zip(axes, metrics):
        for i, name in enumerate(SYSTEMS):
            vals = [_get(rows, name, v)[key] for v in (1, 2)]
            pos = x + (i - 0.5) * (w + 0.02)
            ax.bar(pos, vals, w, color=SYSTEM_COLOUR[name], label=name, zorder=2)
            for p, val in zip(pos, vals):
                ax.text(p, val, f"{val}", ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks(x, ["v1", "v2"])
        ax.set_title(label, loc="left", fontsize=10)
        ax.margins(y=0.15)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper right", ncol=2, bbox_to_anchor=(0.99, 1.0))
    fig.suptitle("Size and complexity of each knowledge base", x=0.01, y=1.0,
                 ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(RESULTS / "fig4_knowledge_cost.png")
    plt.close(fig)


def fig_ablation(rows, perf_rows):
    groups = ["Hierarchical", "Hierarchical + default", "Flat"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    rng = np.random.default_rng(0)
    for ax, key, label in [(axes[0], "macro_f1", "Test macro-F1"),
                           (axes[1], "silently_wrong_pct", "Silently wrong answers (%)")]:
        for i, name in enumerate(groups):
            vals = np.array([r[key] for r in rows if r["system"] == name])
            jitter = rng.uniform(-0.13, 0.13, len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=34, color=SYSTEM_COLOUR[name],
                       edgecolor="white", linewidth=1.2, zorder=3)
            ax.hlines(vals.mean(), i - 0.25, i + 0.25, color=INK, lw=1.5, zorder=4)
        ax.set_xticks(range(3), ["Hierarchical", "Hierarchical\n+ default", "Flat"])
        ax.set_xlim(-0.6, 2.6)
        ax.set_title(label, loc="left", fontsize=10)
    fig.suptitle("Robustness: each dot = test result with one rule removed (version 2); bar = mean",
                 x=0.01, y=1.0, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(RESULTS / "fig5_rule_ablation.png")
    plt.close(fig)


if __name__ == "__main__":
    main()
