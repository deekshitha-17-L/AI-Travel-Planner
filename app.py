import re
import streamlit as st
import pandas as pd

from url_processor import extract_url_content
from recommender import POIRecommender, parse_trip_request
from itinerary import build_itinerary

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
:root { --navy:#0f172a; --purple:#5b4df7; --soft:#f5f7fb; --muted:#64748b; --line:#e2e8f0; }
.block-container { padding: 1.4rem 2.2rem 3.5rem; max-width: 1500px; }
.hero { background:linear-gradient(135deg,#111c33 0%,#3b2d86 100%); color:white; padding:2.25rem 2.6rem; border-radius:0 0 30px 30px; margin:0 0 1.5rem; }
.hero h1 { font-size:3.25rem; line-height:1.05; margin:0 0 .55rem; letter-spacing:-.04em; }
.hero p { color:#e5e7eb; margin:.25rem 0; font-size:1.02rem; line-height:1.65; max-width:950px; }
.badges span { display:inline-block; background:rgba(255,255,255,.12); padding:.42rem .78rem; border-radius:999px; margin:.7rem .35rem 0 0; font-size:.82rem; }
.section-title { margin:1.7rem 0 .35rem; color:#17213a; font-size:1.75rem; font-weight:850; letter-spacing:-.02em; }
.section-subtitle { color:#64748b; margin-bottom:1rem; }
.metric-box { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.05rem 1.15rem; min-height:92px; box-shadow:0 5px 18px rgba(15,23,42,.045); }
.metric-label { color:#8290a6; font-size:.70rem; font-weight:850; letter-spacing:.1em; text-transform:uppercase; }
.metric-value { color:#17213a; font-size:1.28rem; font-weight:850; margin-top:.35rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.place-card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.15rem 1.2rem; min-height:245px; height:100%; box-shadow:0 5px 18px rgba(15,23,42,.045); }
.place-top { display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }
.place-no { color:#5b4df7; background:#eef0ff; border-radius:10px; padding:.35rem .55rem; font-weight:850; font-size:.78rem; }
.place-title { font-size:1.12rem; line-height:1.3; font-weight:800; color:#17213a; margin:.55rem 0 .35rem; }
.match { color:#5146e5; font-weight:850; white-space:nowrap; }
.place-cat { color:#5146e5; font-size:.82rem; font-weight:700; }
.small { color:#64748b; font-size:.84rem; line-height:1.5; }
.place-meta { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; margin-top:.9rem; }
.meta-pill { background:#f8fafc; border:1px solid #edf1f6; border-radius:10px; padding:.55rem .65rem; font-size:.84rem; color:#334155; }
.budget-card { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.15rem 1.2rem; box-shadow:0 5px 18px rgba(15,23,42,.045); }
.budget-row { display:flex; justify-content:space-between; align-items:center; gap:1rem; padding:.7rem 0; border-bottom:1px solid #eef2f7; }
.budget-row:last-child { border-bottom:0; }
.budget-name { font-weight:700; color:#243047; }
.budget-amount { font-weight:850; color:#17213a; white-space:nowrap; }
.budget-note { color:#64748b; font-size:.78rem; }
.day-box { background:#fff; border:1px solid var(--line); border-radius:18px; padding:1.2rem 1.3rem; margin-bottom:1rem; box-shadow:0 5px 18px rgba(15,23,42,.04); }
.day-head { display:flex; justify-content:space-between; gap:1rem; align-items:center; font-size:1.18rem; font-weight:850; color:#17213a; }
.item { border-left:4px solid #5b4df7; padding:.72rem .9rem; margin:.7rem 0 0; background:#f8f9fd; border-radius:0 12px 12px 0; }
.check-card { background:#f8fafc; border:1px solid var(--line); border-radius:14px; padding:.8rem 1rem; }
div[data-testid="stSidebar"] { background:#111827; }
div[data-testid="stSidebar"] * { color:#f3f4f6; }
div[data-testid="stSidebar"] label { color:#e5e7eb !important; }
div[data-testid="stSidebar"] .stTextArea textarea, div[data-testid="stSidebar"] .stTextInput input, div[data-testid="stSidebar"] .stNumberInput input { border-radius:12px; }
.stButton > button { border-radius:12px; font-weight:750; min-height:44px; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); padding:.8rem; border-radius:14px; }
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_recommender():
    return POIRecommender("dataset/poi_data.csv")

try:
    recommender = load_recommender()
except Exception as exc:
    st.error("The recommendation model could not be loaded.")
    st.code(str(exc))
    st.stop()

# ---------- Helpers ----------
def parse_person_count(text: str) -> int:
    text = str(text or "").lower()
    patterns = [
        r"(?:for|with)\s+(\d{1,2})\s*(?:people|persons|person|travellers|travelers|adults)",
        r"(\d{1,2})\s*(?:people|persons|person|travellers|travelers|adults)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return max(1, min(20, int(match.group(1))))
    return 1


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## ✈️ AI Travel Planner")
    st.caption("Personalized trip planning")
    st.markdown("---")

    st.markdown("### Describe your trip")
    request_text = st.text_area(
        "Tell the planner what you want",
        value="I love travelling to Hyderabad for 3 days with a budget of ₹10000. I enjoy history, food and shopping.",
        height=125,
        label_visibility="collapsed",
    )

    surprise = st.button("🎲 Surprise Me", use_container_width=True)

    st.markdown("### Trip controls")

    parsed = parse_trip_request(request_text)
    known_cities = recommender.available_destinations()

    destination_options = sorted(set(known_cities + ([parsed["destination"]] if parsed["destination"] else [])))
    default_destination = parsed["destination"] if parsed["destination"] in destination_options else (destination_options[0] if destination_options else "")

    destination = st.text_input("Destination", value=default_destination, placeholder="Any city or destination")

    days_default = int(parsed["days"] or 3)
    days = st.number_input("Number of days", 1, 14, min(days_default, 14))

    budget_default = int(parsed["budget"] or 10000)
    budget = st.number_input("Total trip budget (₹)", 500, 500000, budget_default, step=500)

    persons_default = parse_person_count(request_text)
    persons = st.number_input(
        "Number of persons travelling",
        min_value=1,
        max_value=20,
        value=persons_default,
        step=1,
    )

    interests_default = parsed["interests"] or ["Sightseeing"]
    interest_choices = [
        "History","Culture","Architecture","Food","Nature","Beach","Adventure",
        "Shopping","Relaxation","Family","Art","Science","Entertainment",
        "Nightlife","Photography","Spiritual","Wildlife","Local Experiences"
    ]
    interests = st.multiselect(
        "Interests",
        interest_choices,
        default=[x for x in interests_default if x in interest_choices],
    )

    travel_style = st.selectbox("Travel pace", ["Relaxed", "Balanced", "Packed"], index=1)

    url = st.text_input(
        "Reference travel URL (optional)",
        placeholder="https://example.com/travel-guide",
    )

    generate = st.button("✨ Create My Trip", type="primary", use_container_width=True)

if surprise:
    st.session_state["surprise_text"] = "Plan a 3-day cultural and food trip with a comfortable budget."
    st.rerun()

# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <h1>✈️ AI Travel Planner</h1>
  <p>Describe your trip in normal language. The planner understands your destination, interests,
  duration and budget, then finds relevant places and builds a practical day-wise itinerary.</p>
  <div class="badges">
    <span>🌍 Any destination</span>
    <span>💬 Natural-language input</span>
    <span>🧠 Semantic recommendations</span>
    <span>🗺️ Dynamic places</span>
    <span>📅 Day-wise planning</span>
  </div>
</div>
""", unsafe_allow_html=True)

if not generate:
    st.info("Start by describing your trip on the left, then click **Create My Trip**.")
    st.markdown("### Example requests")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card"><b>Weekend</b><br><span class="small">“I want a relaxed 2-day trip to Jaipur for history and food.”</span></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card"><b>Family</b><br><span class="small">“Plan 4 days in Hyderabad with family, shopping and local food.”</span></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card"><b>Free-form</b><br><span class="small">“I love beaches and photography. Take me somewhere for 3 days.”</span></div>', unsafe_allow_html=True)
    st.stop()

if not destination.strip():
    st.error("Please enter a destination or mention one in your sentence.")
    st.stop()

interest_text = ", ".join(interests) if interests else "general sightseeing"

# ---------- URL ----------
url_text = ""
if url.strip():
    with st.spinner("Reading the reference travel page..."):
        url_text, url_status = extract_url_content(url)
    if url_text:
        st.success(url_status)
    else:
        st.warning(url_status)

# ---------- Recommendation ----------
persons = int(persons)
total_budget = float(budget)
budget_per_person = total_budget / persons

with st.spinner(f"Finding the best places in {destination}..."):
    recommendations, source_info = recommender.recommend(
        destination=destination,
        interests=interest_text,
        budget=budget_per_person,
        days=days,
        travel_style=travel_style,
        url_text=url_text,
        top_k=12,
    )

if recommendations.empty:
    st.error(
        f"I couldn't find enough place information for **{destination}**. "
        "Check the spelling or try a nearby city."
    )
    st.stop()

# ---------- Summary ----------
st.markdown('<div class="section-title">Your personalized trip</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-subtitle">{len(recommendations)} places considered · {source_info} · Interests: {interest_text}</div>',
    unsafe_allow_html=True,
)

m1, m2, m3, m4, m5 = st.columns(5, gap="medium")
metrics = [
    ("DESTINATION", destination.title()),
    ("DURATION", f"{int(days)} days"),
    ("PERSONS", str(persons)),
    ("TOTAL BUDGET", f"₹{total_budget:,.0f}"),
    ("TRAVEL PACE", travel_style),
]
for col, (label, value) in zip([m1,m2,m3,m4,m5], metrics):
    with col:
        st.markdown(f'<div class="metric-box"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>', unsafe_allow_html=True)

st.markdown("### 📍 Recommended places")

cols = st.columns(3, gap="medium")
for i, (_, row) in enumerate(recommendations.iterrows()):
    rating = row.get("rating")
    rating_text = f"⭐ {float(rating):.1f}" if pd.notna(rating) and float(rating) > 0 else "⭐ Rating unavailable"
    cost = float(row.get("cost", 0) or 0)
    cost_text = "Free / ₹0 per person" if cost <= 0 else f"Est. ₹{cost:,.0f} / person"
    visit_text = f"⏱ {float(row['visit_time_hours']):.1f} hrs"
    match = max(0, min(99, round(float(row.get("similarity", 0)) * 100)))
    description = str(row.get("description", "")).strip() or "A place worth considering for this trip."

    with cols[i % 3]:
        st.markdown(
            f"""
            <div class="place-card">
              <div class="place-top"><span class="place-no">{i+1:02d}</span><span class="match">{match}% match</span></div>
              <div class="place-title">{row['poi']}</div>
              <div class="place-cat">{row['category']}</div>
              <div class="small" style="margin-top:.65rem;min-height:50px;">{description}</div>
              <div class="place-meta">
                <div class="meta-pill">{rating_text}</div>
                <div class="meta-pill">{visit_text}</div>
                <div class="meta-pill">💰 {cost_text}</div>
                <div class="meta-pill">📍 {row.get('source', 'Project dataset')}</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("### 🗓️ Day-wise itinerary")

itinerary, checks = build_itinerary(
    recommendations,
    days=int(days),
    budget=budget_per_person,
    travel_style=travel_style,
)

for day in itinerary:
    items_html = ""
    for item in day["items"]:
        items_html += (
            f"<div class='item'><b>{item['time']} · {item['poi']}</b><br>"
            f"<span class='small'>{item['category']} · {item['cost_display']} · "
            f"{item['visit_hours']:.1f} hrs</span><br>"
            f"<span class='small'>{item['reason']}</span></div>"
        )
    if not items_html:
        items_html = "<div class='small'>No place allocated for this day.</div>"
    st.markdown(
        f"<div class='day-box'><div class='day-head'>DAY {day['day']} · {day['theme']}</div>"
        f"<div class='small'>Planned visiting time: {day['hours']:.1f} hrs</div>{items_html}</div>",
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-title">Budget plan</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Your total group budget is divided into practical spending categories. Place-entry estimates are multiplied by the number of travellers.</div>', unsafe_allow_html=True)

# Practical budget allocation for a student/traveller planning estimate.
budget_split = {
    "Accommodation": 0.35,
    "Food & drinks": 0.25,
    "Local transport": 0.15,
    "Activities & entry fees": 0.15,
    "Shopping / misc.": 0.10,
}
planned_place_spend_per_person = float(checks["budget_used"])
planned_place_spend = planned_place_spend_per_person * persons
allocated = {k: total_budget * pct for k, pct in budget_split.items()}

b1, b2 = st.columns([1.15, 1], gap="large")
with b1:
    st.markdown('<div class="budget-card"><b>Estimated trip budget split</b>', unsafe_allow_html=True)
    for name, amount in allocated.items():
        pct = budget_split[name] * 100
        st.markdown(
            f'<div class="budget-row"><div><div class="budget-name">{name}</div><div class="budget-note">{pct:.0f}% of total budget</div></div><div class="budget-amount">₹{amount:,.0f}</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

with b2:
    st.markdown('<div class="budget-card"><b>Plan estimate</b>', unsafe_allow_html=True)
    st.markdown(f'<div class="budget-row"><div class="budget-name">Travellers</div><div class="budget-amount">{persons} persons</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="budget-row"><div class="budget-name">Total group budget</div><div class="budget-amount">₹{total_budget:,.0f}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="budget-row"><div class="budget-name">Budget per person</div><div class="budget-amount">₹{budget_per_person:,.0f}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="budget-row"><div class="budget-name">Places / entry estimate (group)</div><div class="budget-amount">₹{planned_place_spend:,.0f}</div></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="budget-row"><div class="budget-name">Unallocated group balance</div><div class="budget-amount">₹{max(0, total_budget-planned_place_spend):,.0f}</div></div>', unsafe_allow_html=True)
    st.markdown('<div class="budget-note" style="margin-top:.8rem;">POI entry estimates are treated as per-person costs. Accommodation, food and transport are planning allocations and are not booking prices.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("#### Trip checks")
c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
c1.metric("Persons", persons)
c2.metric("Estimated place spend", f"₹{planned_place_spend:,.0f}")
c3.metric("Budget remaining", f"₹{max(0, total_budget-planned_place_spend):,.0f}")
c4.metric("Places planned", checks["selected_places"])
c5.metric("Time check", "PASS" if checks["time_ok"] else "REVIEW")

if planned_place_spend <= total_budget:
    st.success("✓ The planned place/entry spending for the group stays within your total budget.")
else:
    st.warning("The planned group place/entry spending is above your total budget. Reduce places, add budget, or increase the number of days carefully.")

st.markdown("### 🧠 How the planner works")
st.markdown(
    """
    **1. Understand** → extracts destination, duration, budget and interests from normal language.  
    **2. Budget by group** → uses the number of travelling persons to calculate per-person and group spending.  
    **3. Expand** → combines your project dataset with dynamically discovered OpenStreetMap places.  
    **4. Recommend** → Sentence Transformer embeddings + cosine similarity rank places by semantic relevance.  
    **5. Estimate** → missing cost and visit-time fields are estimated by place category; unknown ratings are shown as unavailable, never as fake zeroes.  
    **5. Plan** → places are distributed across days with time buffers and budget checks.  
    """
)

if url_text:
    with st.expander("Reference information used"):
        st.write(url_text[:3000])

st.divider()
st.caption("AI Travel Planner · Academic project prototype · External place details and estimates should be verified before booking.")
