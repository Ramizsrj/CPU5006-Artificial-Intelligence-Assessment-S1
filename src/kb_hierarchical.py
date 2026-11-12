# System A - hierarchical rule base.
# Inputs -> PRICE and COMFORT -> TECH -> final class, based on the concept
# structure in car.names (Bohanec and Rajkovic, 1988).
# v1 = rules from domain reasoning only, v2 = after one round of fixes using
# the dev set errors (test set not used).
# AI use: rules drafted with help from Claude (Anthropic), see README.

from engine import Rule

HIGH_COST = {"vhigh", "high"}
FOUR_PLUS = {"4", "more"}


def _v1():
    return [
        # PRICE (inputs: buying, maint)
        Rule("P1", {"buying": {"vhigh"}, "maint": HIGH_COST}, ("price", "very_high"), 3,
             "very high purchase price combined with high running costs is unaffordable"),
        Rule("P2", {"buying": {"high"}, "maint": {"vhigh"}}, ("price", "very_high"), 3,
             "high purchase price with very high maintenance is unaffordable"),
        Rule("P3", {"buying": HIGH_COST}, ("price", "high"), 2,
             "an expensive purchase price makes the overall price high"),
        Rule("P4", {"maint": HIGH_COST}, ("price", "high"), 2,
             "expensive maintenance makes the overall price high"),
        Rule("P5", {"buying": {"med"}}, ("price", "medium"), 1, "moderate purchase price"),
        Rule("P6", {"maint": {"med"}}, ("price", "medium"), 1, "moderate maintenance"),
        Rule("P7", {"buying": {"low"}, "maint": {"low"}}, ("price", "low"), 1,
             "cheap to buy and cheap to run"),

        # COMFORT (inputs: doors, persons, lug_boot)
        Rule("C1", {"persons": {"2"}}, ("comfort", "low"), 2,
             "a two-seater cannot serve as a general-purpose car"),
        Rule("C2", {"doors": {"2"}, "lug_boot": {"small"}}, ("comfort", "low"), 2,
             "two doors and a small boot make the car impractical"),
        Rule("C3", {"persons": FOUR_PLUS, "lug_boot": {"med", "big"}, "doors": {"4", "5more"}},
             ("comfort", "high"), 1, "enough seats, doors and luggage space"),
        Rule("C4", {"persons": FOUR_PLUS}, ("comfort", "medium"), 0,
             "carries a family but has some practical limitation"),

        # TECH (inputs: COMFORT, safety)
        Rule("T1", {"safety": {"low"}}, ("tech", "unacceptable"), 2,
             "low safety is a veto criterion"),
        Rule("T2", {"comfort": {"low"}}, ("tech", "unacceptable"), 2,
             "insufficient comfort is a veto criterion"),
        Rule("T3", {"safety": {"high"}, "comfort": {"high"}}, ("tech", "excellent"), 1),
        Rule("T4", {"safety": {"high"}, "comfort": {"medium"}}, ("tech", "good"), 1),
        Rule("T5", {"safety": {"med"}, "comfort": {"high"}}, ("tech", "good"), 1),
        Rule("T6", {"safety": {"med"}, "comfort": {"medium"}}, ("tech", "acceptable"), 1),

        # CAR (inputs: PRICE, TECH)
        Rule("K1", {"tech": {"unacceptable"}}, ("class", "unacc"), 3,
             "technically unacceptable cars are rejected whatever the price"),
        Rule("K2", {"price": {"very_high"}}, ("class", "unacc"), 3,
             "unaffordable cars are rejected whatever the technology"),
        Rule("K3", {"tech": {"excellent"}, "price": {"low", "medium"}}, ("class", "vgood"), 1),
        Rule("K4", {"tech": {"excellent"}, "price": {"high"}}, ("class", "good"), 1),
        Rule("K5", {"tech": {"good"}, "price": {"low", "medium"}}, ("class", "good"), 1),
        Rule("K6", {"tech": {"good"}, "price": {"high"}}, ("class", "acc"), 1),
        Rule("K7", {"tech": {"acceptable"}}, ("class", "acc"), 1),
    ]


# v2 changes: (rule replaced or None if new, new rule, why)
REFINEMENTS = [
    ("C2", Rule("C2", {"doors": {"2"}, "lug_boot": {"small"}, "persons": {"more"}},
                ("comfort", "low"), 2,
                "two doors and a small boot are only impractical when carrying 5+ people"),
     "Dev errors: 2-door/small-boot/4-seat cars were rated acc/good, not unacc; "
     "the veto was too broad."),
    ("C3", Rule("C3", {"persons": FOUR_PLUS, "lug_boot": {"big"}}, ("comfort", "high"), 1,
                "a big boot gives high comfort regardless of the number of doors"),
     "Dev errors: big-boot 2/3-door cars were under-rated (vgood->good, good->acc)."),
    (None, Rule("C5", {"persons": FOUR_PLUS, "lug_boot": {"med"}, "doors": {"4", "5more"}},
                ("comfort", "high"), 1, "medium boot is enough when access is easy (4+ doors)"),
     "Keeps the medium-boot part of the original C3."),
    ("K4", Rule("K4", {"tech": {"excellent"}, "price": {"high"}}, ("class", "acc"), 1,
                "an excellent car at a high price is only acceptable"),
     "Dev errors: 30 acc cars predicted good via K4."),
    (None, Rule("K8", {"tech": {"acceptable"}, "price": {"high"}}, ("class", "unacc"), 2,
                "merely acceptable technology does not justify a high price"),
     "Dev errors: 34 unacc cars predicted acc via K7."),
]


def build_rules(version=2):
    rules = _v1()
    if version >= 2:
        for target, new_rule, _reason in REFINEMENTS:
            if target is None:
                rules.append(new_rule)
            else:
                rules = [new_rule if r.name == target else r for r in rules]
    return rules
