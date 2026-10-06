# AI Travel Planner — Updated Version

This version keeps the project dataset and adds a dynamic discovery layer.

## What changed

- Clean, aligned Streamlit UI.
- Main heading is simply **AI Travel Planner**.
- Removed Review 2 / prototype wording from the visible app.
- Natural-language trip input:
  - "I love travelling to Hyderabad for 3 days"
  - "Plan a 4 day family trip to Jaipur with food and history"
  - "Take me to Goa for 3 days for beaches and photography"
- Destination is no longer limited to the CSV city list.
- Project CSV is still used when it contains the destination.
- OpenStreetMap/Nominatim dynamically supplements the CSV with additional places.
- Missing costs are estimated instead of being shown as ₹0.
- Missing visit times are estimated by category instead of showing tiny/default values.
- Missing ratings are shown as **Rating unavailable**, never as fake ratings.
- Each place gets a readable description.
- Recommendations use Sentence Transformer semantic similarity.
- Itinerary uses budget + daily time constraints.
- The UI shows budget used, budget left, places planned and time validation.

## Run

```bash
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The first run downloads the Sentence Transformer model.

## Important

OpenStreetMap is used only for dynamic place discovery. It does not guarantee ticket prices, ratings, opening hours, or visit duration. The app therefore labels those values as estimates/unavailable and they should be verified before real travel.

Keep your existing `dataset/poi_data.csv` and `dataset/evaluation_data.csv` files in the `dataset` folder.
