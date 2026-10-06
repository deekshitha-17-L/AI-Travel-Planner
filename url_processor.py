import requests
from bs4 import BeautifulSoup

def extract_url_content(url: str, max_chars: int = 8000):
    """Fetch readable text from a public webpage."""
    url = str(url or "").strip()
    if not url:
        return "", "No URL provided."

    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 AI-Travel-Planner/1.0"},
            timeout=12,
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        text = " ".join(text.split())[:max_chars]
        if not text:
            return "", "The page did not contain readable text."

        return text, "Reference information loaded successfully."
    except Exception as exc:
        return "", f"Could not read the URL: {exc}"
