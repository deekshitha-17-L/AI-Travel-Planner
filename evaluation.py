import pandas as pd
from recommender import POIRecommender

def run_evaluation(
    evaluation_csv="dataset/evaluation_data.csv",
    poi_csv="dataset/poi_data.csv",
    k=5,
):
    cases = pd.read_csv(evaluation_csv)
    recommender = POIRecommender(poi_csv)
    hits = 0
    total_expected = 0
    matched_expected = 0
    rows = []

    for _, case in cases.iterrows():
        expected = [x.strip().lower() for x in str(case["expected_pois"]).split(";") if x.strip()]
        result, _ = recommender.recommend(
            destination=case["destination"],
            interests=case["interests"],
            budget=case["budget"],
            days=case["days"],
            travel_style="Balanced",
            top_k=k,
        )
        predicted = result["poi"].str.lower().tolist() if not result.empty else []
        matched = len(set(expected) & set(predicted))
        if matched > 0:
            hits += 1
        matched_expected += matched
        total_expected += len(expected)
        rows.append({
            "destination": case["destination"],
            "expected": "; ".join(expected),
            "predicted_top_k": "; ".join(predicted),
            "matched": matched,
        })

    n = len(cases)
    return {
        "cases": n,
        "hit_rate_at_k": hits / n if n else 0,
        "precision_at_k": matched_expected / (n * k) if n else 0,
        "recall": matched_expected / total_expected if total_expected else 0,
        "details": pd.DataFrame(rows),
    }

if __name__ == "__main__":
    result = run_evaluation()
    print("Test cases:", result["cases"])
    print(f"Hit@5: {result['hit_rate_at_k']:.2%}")
    print(f"Precision@5: {result['precision_at_k']:.2%}")
    print(f"Recall: {result['recall']:.2%}")
    print(result["details"])
