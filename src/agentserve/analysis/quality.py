"""Development-only selection rule; fresh acceptance must never participate."""


def select_profile(runs, profile_order):
    grouped = {
        name: {r["dataset"]: r for r in runs if r["profile"] == name} for name in profile_order
    }
    baseline = grouped["baseline"]
    eligible = [
        name
        for name, group in grouped.items()
        if all(
            group[d]["quality"].get("correct", 0) >= baseline[d]["quality"].get("correct", 0)
            for d in ["demo", "development", "business"]
        )
    ]
    return max(
        eligible, key=lambda name: sum(r["completed_correct"] for r in grouped[name].values())
    )
