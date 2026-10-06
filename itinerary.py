import math
import pandas as pd

PACE = {
    "Relaxed": {"max_hours": 6.0, "places_per_day": 2},
    "Balanced": {"max_hours": 7.5, "places_per_day": 3},
    "Packed": {"max_hours": 9.0, "places_per_day": 4},
}

THEMES = ["Explore & Discover", "Culture & History", "Food & Local Life", "Nature & Relaxation"]

def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(x) for x in [lat1, lon1, lat2, lon2]):
        return None
    R = 6371.0
    p1, p2 = math.radians(float(lat1)), math.radians(float(lat2))
    dp = math.radians(float(lat2) - float(lat1))
    dl = math.radians(float(lon2) - float(lon1))
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _format_cost(value):
    return f"₹{float(value):,.0f}" if value is not None else "Estimated spend unavailable"

def build_itinerary(recommendations: pd.DataFrame, days: int, budget: float, travel_style="Balanced"):
    if recommendations.empty:
        return [], {
            "budget_used": 0, "budget_ok": False, "days_ok": False,
            "time_ok": False, "selected_places": 0
        }

    days = max(1, int(days))
    pace = PACE.get(travel_style, PACE["Balanced"])

    candidates = recommendations.copy()
    candidates["cost_num"] = pd.to_numeric(candidates["cost"], errors="coerce").fillna(150)
    candidates["visit_num"] = pd.to_numeric(candidates["visit_time_hours"], errors="coerce").fillna(1.5).clip(0.5, 4.0)

    selected = []
    spent = 0.0
    day_hours = [0.0] * days

    # Greedy selection with per-day time and total budget constraints.
    for _, row in candidates.iterrows():
        cost = float(row["cost_num"])
        visit = float(row["visit_num"])

        possible_days = [
            d for d in range(days)
            if day_hours[d] + visit <= pace["max_hours"]
            and spent + cost <= float(budget)
        ]
        if not possible_days:
            continue

        target_day = min(possible_days, key=lambda d: day_hours[d])
        selected.append((target_day, row))
        day_hours[target_day] += visit
        spent += cost

        if len(selected) >= days * pace["places_per_day"]:
            break

    # Always put at least one place in the plan when possible.
    if not selected:
        row = candidates.iloc[0]
        selected = [(0, row)]
        spent = float(row["cost_num"])
        day_hours[0] = float(row["visit_num"])

    buckets = [[] for _ in range(days)]
    for day_idx, row in selected:
        buckets[day_idx].append(row)

    itinerary = []
    for day_no, bucket in enumerate(buckets, start=1):
        current_time = 9.0
        items = []

        for row in bucket:
            visit = float(row["visit_num"])
            cost = float(row["cost_num"])

            start_h = int(current_time)
            start_m = int(round((current_time - start_h) * 60)) % 60
            end = current_time + visit
            end_h = int(end)
            end_m = int(round((end - end_h) * 60)) % 60

            category = str(row.get("category", "Attraction"))
            score = float(row.get("final_score", row.get("similarity", 0)))

            items.append({
                "poi": row["poi"],
                "category": category,
                "time": f"{start_h:02d}:{start_m:02d} – {end_h:02d}:{end_m:02d}",
                "cost": cost,
                "cost_display": _format_cost(cost),
                "visit_hours": visit,
                "reason": f"Selected because it matches your interests (relevance score {score:.2f}).",
            })

            current_time = end + 0.5

        itinerary.append({
            "day": day_no,
            "theme": THEMES[(day_no - 1) % len(THEMES)],
            "items": items,
            "hours": sum(x["visit_hours"] for x in items),
        })

    checks = {
        "budget_used": spent,
        "budget_ok": spent <= float(budget),
        "days_ok": len(itinerary) == days,
        "time_ok": all(day["hours"] <= pace["max_hours"] for day in itinerary),
        "selected_places": len(selected),
    }

    return itinerary, checks
