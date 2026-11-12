# System B - flat rule base.
# Every rule goes straight from the six inputs to the class, with a default
# rule at the end so every car gets an answer.
# v1 = rules from domain reasoning only, v2 = after one round of fixes using
# the dev set errors (test set not used).
# AI use: rules drafted with help from Claude (Anthropic), see README.

from engine import Rule

CHEAP = {"low", "med"}
FOUR_PLUS = {"4", "more"}


def _v1():
    return [
        # rejection rules (highest priority: any one is a veto)
        Rule("F1", {"safety": {"low"}}, ("class", "unacc"), 3, "low safety is unacceptable"),
        Rule("F2", {"persons": {"2"}}, ("class", "unacc"), 3, "two-seaters are unacceptable"),
        Rule("F3", {"buying": {"vhigh"}, "maint": {"vhigh", "high"}}, ("class", "unacc"), 3,
             "unaffordable"),
        Rule("F4", {"buying": {"high"}, "maint": {"vhigh"}}, ("class", "unacc"), 3,
             "unaffordable"),
        Rule("F5", {"doors": {"2"}, "lug_boot": {"small"}}, ("class", "unacc"), 3,
             "impractical body"),

        # positive rules
        Rule("F6", {"safety": {"high"}, "lug_boot": {"big"}, "persons": FOUR_PLUS,
                    "buying": CHEAP, "maint": CHEAP}, ("class", "vgood"), 2,
             "safe, roomy and cheap"),
        Rule("F7", {"safety": {"high"}, "persons": FOUR_PLUS,
                    "buying": CHEAP, "maint": CHEAP}, ("class", "good"), 1,
             "safe and cheap"),
        Rule("F8", {"safety": {"med", "high"}, "persons": FOUR_PLUS}, ("class", "acc"), 0,
             "adequate safety and capacity"),

        # default
        Rule("F0", {}, ("class", "unacc"), -1, "default: reject if nothing else applies"),
    ]


HIGH_COST = {"vhigh", "high"}

# v2 changes: (rule replaced or None if new, new rule, why)
# same ideas as the hierarchical fixes, but without intermediate concepts
# one idea can need several rules
REFINEMENTS = [
    ("F5", Rule("F5", {"doors": {"2"}, "lug_boot": {"small"}, "persons": {"more"}},
                ("class", "unacc"), 3, "impractical body only when carrying 5+ people"),
     "Dev errors: 2-door/small-boot/4-seat cars were acc/good, not unacc."),
    # 'ok car but too expensive' needs 4 flat rules here (1 rule in system A)
    (None, Rule("F9", {"safety": {"med"}, "lug_boot": {"small"}, "buying": HIGH_COST},
                ("class", "unacc"), 2, "adequate car, too expensive to buy"),
     "Dev errors: 34 unacc cars predicted acc via F8."),
    (None, Rule("F10", {"safety": {"med"}, "lug_boot": {"small"}, "maint": HIGH_COST},
                ("class", "unacc"), 2, "adequate car, too expensive to run"), ""),
    (None, Rule("F11", {"safety": {"med"}, "lug_boot": {"med"}, "doors": {"2", "3"},
                        "buying": HIGH_COST}, ("class", "unacc"), 2,
                "adequate car, too expensive to buy"), ""),
    (None, Rule("F12", {"safety": {"med"}, "lug_boot": {"med"}, "doors": {"2", "3"},
                        "maint": HIGH_COST}, ("class", "unacc"), 2,
                "adequate car, too expensive to run"), ""),
    # roomy + cheap + medium safety = good
    (None, Rule("F13", {"safety": {"med"}, "persons": FOUR_PLUS, "lug_boot": {"big"},
                        "buying": CHEAP, "maint": CHEAP}, ("class", "good"), 1,
                "roomy and cheap with adequate safety"),
     "Dev errors: 19 good cars predicted acc via F8."),
    (None, Rule("F14", {"safety": {"med"}, "persons": FOUR_PLUS, "lug_boot": {"med"},
                        "doors": {"4", "5more"}, "buying": CHEAP, "maint": CHEAP},
                ("class", "good"), 1, "comfortable and cheap with adequate safety"), ""),
    # medium boot is fine with 4+ doors
    (None, Rule("F15", {"safety": {"high"}, "persons": FOUR_PLUS, "lug_boot": {"med"},
                        "doors": {"4", "5more"}, "buying": CHEAP, "maint": CHEAP},
                ("class", "vgood"), 2, "safe, comfortable and cheap"),
     "Dev errors: vgood cars with medium boot predicted good via F7."),
]


def build_rules(version=2):
    rules = _v1()
    if version >= 2:
        for target, new_rule, _reason in REFINEMENTS:
            if target is None:
                rules.insert(len(rules) - 1, new_rule)  # default stays last
            else:
                rules = [new_rule if r.name == target else r for r in rules]
    return rules
