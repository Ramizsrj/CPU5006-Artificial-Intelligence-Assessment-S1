# Forward chaining inference engine used by both expert systems.
# AI use: written with help from Claude (Anthropic), see README.
#
# Each cycle: find all rules that match the current facts (conflict set),
# pick one (highest priority, then most conditions, then first in the list),
# fire it and add its conclusion to the facts. Stop when nothing else fires.
# A fact is never overwritten once set (refraction), so it always stops.

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Rule:
    name: str
    conditions: dict      # attribute -> set of allowed values, {} = always true
    conclusion: tuple     # (attribute, value)
    priority: int = 0
    note: str = ""        # reason for the rule, used in explanations

    @property
    def specificity(self):
        return len(self.conditions)

    def matches(self, facts):
        for attr, allowed in self.conditions.items():
            if facts.get(attr) not in allowed:
                return False
        return True

    def __str__(self):
        parts = []
        for attr, vals in self.conditions.items():
            if len(vals) == 1:
                parts.append(f"{attr} = {next(iter(vals))}")
            else:
                parts.append(f"{attr} in {sorted(vals)}")
        lhs = " AND ".join(parts) if parts else "TRUE"
        return f"{self.name}: IF {lhs} THEN {self.conclusion[0]} = {self.conclusion[1]}"


@dataclass
class InferenceResult:
    facts: dict
    trace: list = field(default_factory=list)   # rules fired in order
    cycles: int = 0
    conflicts: int = 0        # times 2+ rules wanted to set the same attribute
    contradictions: int = 0   # ...and they disagreed on the value
    used_default: bool = False


class ForwardChainingEngine:
    def __init__(self, rules, goal):
        self.rules = list(rules)
        self.goal = goal
        self.order = {r.name: i for i, r in enumerate(self.rules)}

    def pick(self, conflict_set):
        return min(conflict_set,
                   key=lambda r: (-r.priority, -r.specificity, self.order[r.name]))

    def infer(self, case):
        facts = dict(case)
        res = InferenceResult(facts=facts)
        while True:
            conflict_set = [r for r in self.rules
                            if r.conclusion[0] not in facts and r.matches(facts)]
            if not conflict_set:
                break
            chosen = self.pick(conflict_set)

            # count conflicts (default rules always match so I leave them out)
            rivals = [r for r in conflict_set
                      if r is not chosen and r.specificity > 0
                      and r.conclusion[0] == chosen.conclusion[0]]
            if rivals:
                res.conflicts += 1
                if any(r.conclusion[1] != chosen.conclusion[1] for r in rivals):
                    res.contradictions += 1

            attr, value = chosen.conclusion
            facts[attr] = value
            res.trace.append(chosen)
            res.cycles += 1
            if attr == self.goal and chosen.specificity == 0:
                res.used_default = True
        return res

    def classify(self, case):
        return self.infer(case).facts.get(self.goal, "unknown")

    def explain(self, case):
        res = self.infer(case)
        lines = [f"Input: {case}"]
        for i, rule in enumerate(res.trace, 1):
            lines.append(f"  {i}. {rule}")
            if rule.note:
                lines.append(f"       because: {rule.note}")
        lines.append(f"Conclusion: {self.goal} = {res.facts.get(self.goal)}")
        return "\n".join(lines)
