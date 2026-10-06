import re
import time
import requests
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

CATEGORY_ESTIMATES = {
    "museum": (100, 2.0),
    "gallery": (100, 1.5),
    "palace": (250, 2.0),
    "castle": (250, 2.0),
    "fort": (100, 2.5),
    "temple": (0, 1.0),
    "church": (0, 0.8),
    "mosque": (0, 0.8),
    "shrine": (0, 1.0),
    "park": (50, 1.5),
    "garden": (50, 1.5),
    "zoo": (250, 3.0),
    "aquarium": (250, 2.5),
    "beach": (100, 2.0),
    "lake": (50, 1.5),
    "viewpoint": (50, 1.0),
    "market": (500, 2.0),
    "shopping": (1000, 2.5),
    "restaurant": (600, 1.5),
    "cafe": (350, 1.0),
    "theatre": (500, 2.0),
    "stadium": (300, 2.0),
    "attraction": (150, 1.5),
    "tourism": (150, 1.5),
}

INTEREST_KEYWORDS = {
    "history": ["history", "historic", "heritage", "fort", "palace", "monument", "archaeology"],
    "culture": ["culture", "cultural", "heritage", "temple", "mosque", "church", "museum"],
    "architecture": ["architecture", "building", "palace", "fort", "monument"],
    "food": ["food", "restaurant", "cafe", "bakery", "market", "street food"],
    "nature": ["nature", "park", "garden", "lake", "forest", "waterfall"],
    "beach": ["beach", "coast", "sea", "shore"],
    "adventure": ["adventure", "trek", "hiking", "camping", "sports"],
    "shopping": ["shopping", "market", "mall", "bazaar"],
    "relaxation": ["relaxation", "spa", "garden", "park", "lake"],
    "family": ["family", "zoo", "aquarium", "park", "museum"],
    "art": ["art", "gallery", "museum", "theatre"],
    "science": ["science", "planetarium", "museum", "technology"],
    "entertainment": ["entertainment", "theatre", "cinema", "stadium"],
    "nightlife": ["nightlife", "bar", "club", "music"],
    "photography": ["photography", "viewpoint", "monument", "lake", "beach"],
    "spiritual": ["temple", "mosque", "church", "shrine"],
    "wildlife": ["zoo", "wildlife", "sanctuary", "bird"],
    "local experiences": ["market", "bazaar", "restaurant", "cafe", "culture"],
}

def parse_trip_request(text: str):
    text = str(text or "").strip()
    lower = text.lower()

    days = None
    m = re.search(r"(\d{1,2})\s*(?:day|days|d)\b", lower)
    if m:
        days = int(m.group(1))

    budget = None
    budget_patterns = [
        r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|l|lakh)?",
        r"budget\s*(?:of|is|:)?\s*₹?\s*([\d,]+(?:\.\d+)?)\s*(k|thousand|l|lakh)?",
    ]
    for pattern in budget_patterns:
        m = re.search(pattern, lower)
        if m:
            budget = _money_value(m.group(1), m.group(2))
            break

    destination = None
    patterns = [
        r"(?:travell?ing|travel|trip|visit|visiting|go|going|vacation|holiday)\s+(?:to|in|at)\s+([A-Za-z][A-Za-z .'-]{1,50}?)(?=\s+(?:for|with|on|under|within|and|,|\.|$))",
        r"\b(?:to|in)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?=\s+(?:for|with|on|under|within|and|,|\.|$))",
    ]
    for pattern in patterns:
        m = re.search(pattern, text, flags=re.I)
        if m:
            destination = m.group(1).strip(" .,-")
            break

    interests = []
    for label in INTEREST_KEYWORDS:
        if any(word in lower for word in label.split()):
            interests.append(label.title())
        elif any(k in lower for k in INTEREST_KEYWORDS[label]):
            interests.append(label.title())

    # Remove overly broad duplicates.
    return {
        "destination": destination,
        "days": days,
        "budget": budget,
        "interests": list(dict.fromkeys(interests))[:6],
    }

def _money_value(number, suffix):
    value = float(str(number).replace(",", ""))
    suffix = (suffix or "").lower()
    if suffix in ("k", "thousand"):
        value *= 1000
    elif suffix in ("l", "lakh"):
        value *= 100000
    return int(value)

class POIRecommender:
    def __init__(self, csv_path="dataset/poi_data.csv"):
        self.model = SentenceTransformer(MODEL_NAME)
        self.df = pd.read_csv(csv_path)
        self.df = self._normalise_dataset(self.df)
        self._build_embeddings()

    def _normalise_dataset(self, df):
        df = df.copy()
        required = {
            "destination": "",
            "poi": "Unknown place",
            "category": "Attraction",
            "rating": None,
            "cost": None,
            "visit_time_hours": None,
            "description": "",
            "latitude": None,
            "longitude": None,
        }
        for col, default in required.items():
            if col not in df.columns:
                df[col] = default

        df["destination"] = df["destination"].fillna("").astype(str)
        df["poi"] = df["poi"].fillna("Unknown place").astype(str)
        df["category"] = df["category"].fillna("Attraction").astype(str)
        df["description"] = df["description"].fillna("").astype(str)

        for col in ["rating", "cost", "visit_time_hours", "latitude", "longitude"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["source"] = "Project dataset"

        # Never turn missing values into misleading zeroes.
        for idx, row in df.iterrows():
            cat = str(row["category"]).lower()
            estimate = self._estimate_for_category(cat)
            if pd.isna(row["cost"]):
                df.at[idx, "cost"] = estimate[0]
            if pd.isna(row["visit_time_hours"]) or float(row["visit_time_hours"] or 0) <= 0:
                df.at[idx, "visit_time_hours"] = estimate[1]

        return df

    def _build_embeddings(self):
        self.df["text"] = (
            self.df["poi"].fillna("") + ". " +
            self.df["category"].fillna("") + ". " +
            self.df["description"].fillna("")
        )
        self.embeddings = self.model.encode(
            self.df["text"].tolist(),
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def available_destinations(self):
        return sorted(x for x in self.df["destination"].dropna().unique().tolist() if str(x).strip())

    def _estimate_for_category(self, category):
        category = str(category).lower()
        for key, value in CATEGORY_ESTIMATES.items():
            if key in category:
                return value
        return (150, 1.5)

    def _estimate_cost_text(self, category, cost):
        c, _ = self._estimate_for_category(category)
        if cost is None or pd.isna(cost):
            return f"Estimated spend ₹{c:,.0f}"
        cost = float(cost)
        # Dataset values are treated as a planning estimate, not an official ticket price.
        return f"Estimated spend ₹{cost:,.0f}"

    def _dynamic_places(self, destination):
        # Free OpenStreetMap/Nominatim discovery. It supplements, rather than replaces,
        # the student's project dataset.
        headers = {"User-Agent": "AI-Travel-Planner-Academic/1.0"}
        queries = [
            f"tourist attractions in {destination}",
            f"museums in {destination}",
            f"parks in {destination}",
            f"historic places in {destination}",
            f"shopping and markets in {destination}",
            f"food places in {destination}",
        ]

        rows = []
        seen = set()
        session = requests.Session()

        for q in queries:
            try:
                response = session.get(
                    NOMINATIM_URL,
                    params={"q": q, "format": "jsonv2", "limit": 8, "addressdetails": 1},
                    headers=headers,
                    timeout=12,
                )
                response.raise_for_status()
                items = response.json()
            except Exception:
                continue

            for item in items:
                name = str(item.get("name") or "").strip()
                if not name:
                    continue
                key = (name.lower(), str(item.get("lat")), str(item.get("lon")))
                if key in seen:
                    continue
                seen.add(key)

                typ = str(item.get("type") or "attraction").replace("_", " ").title()
                category = self._osm_category(item, typ)
                cost, visit = self._estimate_for_category(category)
                address = item.get("display_name", "")
                rows.append({
                    "destination": destination,
                    "poi": name,
                    "category": category,
                    "rating": None,
                    "cost": float(cost),
                    "visit_time_hours": float(visit),
                    "description": self._dynamic_description(name, category, address),
                    "latitude": pd.to_numeric(item.get("lat"), errors="coerce"),
                    "longitude": pd.to_numeric(item.get("lon"), errors="coerce"),
                    "source": "OpenStreetMap",
                })

            # Nominatim requests should be polite.
            time.sleep(1.0)

        return pd.DataFrame(rows)

    def _osm_category(self, item, fallback):
        blob = " ".join([
            str(item.get("type", "")),
            str(item.get("category", "")),
            str(item.get("class", "")),
            str(item.get("display_name", "")),
        ]).lower()

        for key in [
            "museum","gallery","palace","castle","fort","temple","church","mosque",
            "shrine","park","garden","zoo","aquarium","beach","lake","viewpoint",
            "market","shopping","restaurant","cafe","theatre","stadium"
        ]:
            if key in blob:
                return key.title()
        return fallback or "Attraction"

    def _dynamic_description(self, name, category, address):
        place = category.lower()
        return f"{name} is a {place} in {address.split(',')[0] if address else 'the destination'}."

    def recommend(self, destination, interests, budget, days, travel_style="Balanced", url_text="", top_k=12):
        destination = str(destination).strip()
        if not destination:
            return pd.DataFrame(), "No destination"

        # First use the student's dataset if it has the city.
        local = self.df[
            self.df["destination"].str.strip().str.lower() == destination.lower()
        ].copy()

        source_info = "Project dataset"
        dynamic = pd.DataFrame()

        # Always supplement with dynamic discovery when possible.
        try:
            dynamic = self._dynamic_places(destination)
        except Exception:
            dynamic = pd.DataFrame()

        if not dynamic.empty:
            combined = pd.concat([local, dynamic], ignore_index=True, sort=False)
            combined = combined.drop_duplicates(
                subset=["poi", "latitude", "longitude"], keep="first"
            )
            source_info = "Project dataset + OpenStreetMap"
        else:
            combined = local

        if combined.empty:
            return pd.DataFrame(), "No place data found"

        interest_text = interests or "general sightseeing"
        query = (
            f"Destination: {destination}. Interests: {interest_text}. "
            f"Trip duration: {days} days. Budget: {budget} INR. "
            f"Travel style: {travel_style}. {url_text[:2500]}"
        )

        q_embedding = self.model.encode([query], normalize_embeddings=True)
        text = (
            combined["poi"].fillna("").astype(str) + ". " +
            combined["category"].fillna("").astype(str) + ". " +
            combined["description"].fillna("").astype(str)
        )
        embeddings = self.model.encode(text.tolist(), normalize_embeddings=True)
        scores = cosine_similarity(q_embedding, embeddings)[0]
        combined["similarity"] = scores

        # Small keyword boost for explicit interests.
        interest_lower = str(interest_text).lower()
        combined["interest_boost"] = combined.apply(
            lambda r: 0.10 if any(
                k in (str(r["category"]) + " " + str(r["description"])).lower()
                for k in interest_lower.split(",")
                if k.strip()
            ) else 0.0,
            axis=1,
        )
        combined["final_score"] = combined["similarity"] + combined["interest_boost"]

        # Travel pace changes how many places we expose.
        limit = {"Relaxed": 9, "Balanced": 12, "Packed": 15}.get(travel_style, 12)

        combined["budget_fit"] = pd.to_numeric(combined["cost"], errors="coerce").fillna(0) <= float(budget)
        combined = combined.sort_values(
            ["final_score", "budget_fit"],
            ascending=[False, False],
        )

        combined["cost_display"] = combined.apply(
            lambda r: self._estimate_cost_text(r["category"], r["cost"]),
            axis=1
        )

        return combined.head(max(top_k, limit)).reset_index(drop=True).head(limit), source_info
