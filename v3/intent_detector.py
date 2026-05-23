def should_use_live_search(message: str) -> bool:
    msg = (message or "").lower().strip()

    live_search_triggers = [
        "latest",
        "current",
        "today",
        "now",
        "open now",
        "opening time",
        "closing time",
        "timing",
        "time now",
        "entry fee",
        "ticket price",
        "price",
        "weather",
        "temperature",
        "rain",
        "train",
        "bus",
        "flight",
        "news",
        "nearby",
        "near me",
        "hospital nearby",
        "restaurant nearby",
        "hotel nearby",
        "petrol pump",
        "atm nearby",
        "pharmacy nearby",
        "is it open",
        "available now"
    ]

    return any(trigger in msg for trigger in live_search_triggers)


def build_live_search_query(message: str) -> str:
    msg = (message or "").strip()

    kerala_words = [
        "kerala",
        "kochi",
        "ernakulam",
        "munnar",
        "wayanad",
        "calicut",
        "kozhikode",
        "malappuram",
        "trivandrum",
        "thiruvananthapuram",
        "kollam",
        "alappuzha",
        "idukki",
        "thrissur",
        "kannur",
        "kasaragod",
        "palakkad",
        "pathanamthitta",
        "kottayam"
    ]

    if any(word in msg.lower() for word in kerala_words):
        return msg

    return f"{msg} Kerala"