import os
import re
import json
import time
import math
import traceback
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from groq import Groq

from v3.live_search import tavily_live_search
from v3.intent_detector import should_use_live_search, build_live_search_query
from v3.response_builder import build_v3_response


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
PORT = int(os.getenv("PORT", "5000"))

FIREBASE_SERVICE_ACCOUNT = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT",
    "serviceAccountKey.json"
)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

MAX_PLACES_CONTEXT = 10
CACHE_TTL_SECONDS = 60 * 60 * 6  # 6 hours
LOCAL_CACHE_FILE = "places_cache.json"

APP_NAME = "KeralaTour AI Backend"


# ============================================================
# FLASK INIT
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# FIREBASE INIT
# ============================================================
firebase_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")

if not firebase_admin._apps:
    if firebase_json:
        cred_dict = json.loads(firebase_json)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT)

    firebase_admin.initialize_app(cred)

db = firestore.client()
# ============================================================
# GROQ INIT
# ============================================================

groq_client = None

if GROQ_API_KEY:
    groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# GLOBAL CACHE
# ============================================================

PLACES_CACHE: List[Dict[str, Any]] = []
PLACES_CACHE_TIME = 0


# ============================================================
# BASIC UTILS
# ============================================================

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value).strip()


def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def normalize_text(text: str) -> str:
    text = safe_text(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_for_display(text: str, limit: int = 600) -> str:
    text = safe_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit].strip() + "..."
    return text


def split_tags(tags: Any) -> List[str]:
    if isinstance(tags, list):
        return [safe_text(t) for t in tags if safe_text(t)]
    if isinstance(tags, str):
        return [t.strip() for t in tags.split(",") if t.strip()]
    return []


def unique_list(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = normalize_text(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item)
    return result


def bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ["1", "true", "yes", "on"]


def debug_log(label: str, data: Any = None):
    print(f"[{now_iso()}] {label}")
    if data is not None:
        try:
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
        except Exception:
            print(data)


# ============================================================
# DOMAIN DATA
# ============================================================

KERALA_DISTRICTS = [
    "thiruvananthapuram",
    "trivandrum",
    "kollam",
    "pathanamthitta",
    "alappuzha",
    "alleppey",
    "kottayam",
    "idukki",
    "ernakulam",
    "kochi",
    "thrissur",
    "palakkad",
    "malappuram",
    "kozhikode",
    "calicut",
    "wayanad",
    "kannur",
    "kasaragod",
]

DISTRICT_ALIASES = {
    "trivandrum": "Thiruvananthapuram",
    "tvm": "Thiruvananthapuram",
    "alleppey": "Alappuzha",
    "calicut": "Kozhikode",
    "cochin": "Kochi",
    "ernakulam": "Kochi",
}

MOOD_KEYWORDS = {
    "romantic": [
        "romantic",
        "couple",
        "honeymoon",
        "peaceful",
        "sunset",
        "private",
        "calm",
    ],
    "family": [
        "family",
        "kids",
        "children",
        "safe",
        "picnic",
        "parents",
        "baby",
        "elder",
    ],
    "adventure": [
        "adventure",
        "trek",
        "trekking",
        "hiking",
        "forest",
        "wildlife",
        "camping",
        "offroad",
    ],
    "budget": [
        "budget",
        "cheap",
        "low cost",
        "affordable",
        "free",
        "less money",
    ],
    "monsoon": [
        "rain",
        "monsoon",
        "rainy",
        "waterfall",
        "green",
    ],
    "beach": [
        "beach",
        "coastal",
        "sea",
        "sunset",
        "shore",
        "lighthouse",
    ],
    "hill": [
        "hill",
        "hill station",
        "munnar",
        "vagamon",
        "ponmudi",
        "tea",
        "misty",
        "cool",
    ],
    "backwater": [
        "backwater",
        "lake",
        "boating",
        "houseboat",
        "canal",
        "kayaking",
    ],
    "wildlife": [
        "wildlife",
        "forest",
        "sanctuary",
        "national park",
        "elephant",
        "tiger",
        "bird",
    ],
    "heritage": [
        "heritage",
        "temple",
        "fort",
        "palace",
        "museum",
        "church",
        "mosque",
        "culture",
    ],
    "food": [
        "food",
        "eat",
        "restaurant",
        "dish",
        "seafood",
        "sadya",
        "biriyani",
        "local food",
    ],
}

INTENT_KEYWORDS = {
    "travel_time": [
        "how much time",
        "how long",
        "time take",
        "travel time",
        "distance",
        "km",
        "from",
        "reach",
        "route",
        "drive",
        "bus",
        "train",
        "car",
        "bike",
        "scooter",
    ],
    "recommendation": [
        "best",
        "top",
        "trending",
        "popular",
        "highest rated",
        "suggest",
        "recommend",
        "where to go",
        "places",
    ],
    "trip_plan": [
        "plan",
        "trip",
        "itinerary",
        "2 day",
        "3 day",
        "one day",
        "1 day",
        "weekend",
        "schedule",
    ],
    "best_time": [
        "when",
        "best time",
        "season",
        "month",
        "monsoon",
        "summer",
        "winter",
        "visit time",
    ],
    "food": [
        "food",
        "eat",
        "restaurant",
        "local food",
        "dish",
        "breakfast",
        "lunch",
        "dinner",
    ],
    "family_safety": [
        "safe",
        "family",
        "kids",
        "child",
        "children",
        "parents",
        "elder",
        "baby",
    ],
    "compare": [
        "compare",
        "which is better",
        "better",
        "or",
        "vs",
        "versus",
    ],
    "details": [
        "what about",
        "tell me about",
        "details",
        "information",
        "info",
    ],
}


KNOWN_DISTANCES_KM = {
    ("malappuram", "munnar"): 250,
    ("kochi", "munnar"): 130,
    ("ernakulam", "munnar"): 130,
    ("kozhikode", "munnar"): 280,
    ("calicut", "munnar"): 280,
    ("thrissur", "munnar"): 150,
    ("trivandrum", "munnar"): 280,
    ("thiruvananthapuram", "munnar"): 280,
    ("palakkad", "munnar"): 175,

    ("malappuram", "wayanad"): 120,
    ("kochi", "wayanad"): 270,
    ("ernakulam", "wayanad"): 270,
    ("calicut", "wayanad"): 90,
    ("kozhikode", "wayanad"): 90,
    ("thrissur", "wayanad"): 230,

    ("kochi", "alleppey"): 55,
    ("ernakulam", "alleppey"): 55,
    ("malappuram", "alleppey"): 190,
    ("kochi", "alappuzha"): 55,
    ("malappuram", "alappuzha"): 190,

    ("kochi", "varkala"): 170,
    ("trivandrum", "varkala"): 45,
    ("thiruvananthapuram", "varkala"): 45,
    ("malappuram", "varkala"): 330,

    ("kochi", "thekkady"): 160,
    ("malappuram", "thekkady"): 280,
    ("trivandrum", "thekkady"): 220,

    ("kochi", "vagamon"): 100,
    ("malappuram", "vagamon"): 210,

    ("kochi", "athirappilly"): 70,
    ("thrissur", "athirappilly"): 60,
    ("malappuram", "athirappilly"): 160,
}


# ============================================================
# FIRESTORE SERVICE
# ============================================================

def normalize_place(doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    description = safe_text(
        data.get("ai_description") or data.get("description"),
        "A beautiful destination in Kerala."
    )

    name = safe_text(data.get("name"), "Unknown Place")
    region = safe_text(data.get("region"), "Kerala")
    tags = split_tags(data.get("tags"))

    search_blob = " ".join([
        doc_id,
        name,
        region,
        safe_text(data.get("distance")),
        safe_text(data.get("bestTime")),
        description,
        " ".join(tags),
    ])

    return {
        "id": doc_id,
        "name": name,
        "region": region,
        "distance": safe_text(data.get("distance"), ""),
        "imageUrl": safe_text(data.get("imageUrl"), ""),
        "rating": safe_float(data.get("rating"), 0.0),
        "userRatings": safe_int(data.get("userRatings"), 0),
        "ratingTotal": safe_float(data.get("ratingTotal"), 0.0),
        "bestTime": safe_text(data.get("bestTime"), "Any time"),
        "description": description,
        "raw_description": safe_text(data.get("description"), ""),
        "ai_description": safe_text(data.get("ai_description"), ""),
        "tags": tags,
        "source": safe_text(data.get("source"), ""),
        "search_blob": normalize_text(search_blob),
    }


def load_places_from_firestore(force: bool = False) -> List[Dict[str, Any]]:
    global PLACES_CACHE, PLACES_CACHE_TIME

    current_time = time.time()

    cache_valid = (
        PLACES_CACHE
        and not force
        and (current_time - PLACES_CACHE_TIME) < CACHE_TTL_SECONDS
    )

    if cache_valid:
        return PLACES_CACHE

    # If memory cache is empty, try local cache first.
    # This prevents Firestore quota problems during testing.
    if not PLACES_CACHE and not force:
        local_places = load_places_from_local_cache()

        if local_places:
            PLACES_CACHE = local_places
            PLACES_CACHE_TIME = current_time

            debug_log("Using local cache instead of Firestore", {
                "count": len(PLACES_CACHE),
            })

            return PLACES_CACHE

    places = []

    try:
        docs = db.collection("places").stream()

        for doc in docs:
            data = doc.to_dict() or {}
            places.append(normalize_place(doc.id, data))

        PLACES_CACHE = places
        PLACES_CACHE_TIME = current_time

        save_places_to_local_cache(PLACES_CACHE)

        debug_log("Places cache refreshed from Firestore", {
            "count": len(PLACES_CACHE),
            "cache_time": PLACES_CACHE_TIME,
        })

        return PLACES_CACHE

    except ResourceExhausted as e:
        debug_log("Firestore quota exceeded", str(e))

        # 1. Use memory cache if available
        if PLACES_CACHE:
            debug_log("Using memory cache after Firestore quota exceeded", {
                "count": len(PLACES_CACHE),
            })
            return PLACES_CACHE

        # 2. Use local file cache if available
        local_places = load_places_from_local_cache()

        if local_places:
            PLACES_CACHE = local_places
            PLACES_CACHE_TIME = current_time

            debug_log("Using local file cache after Firestore quota exceeded", {
                "count": len(PLACES_CACHE),
            })

            return PLACES_CACHE

        # 3. No cache available
        raise RuntimeError(
            "Firestore quota exceeded and no local cache exists yet. "
            "Open /api/places once tomorrow or after quota reset to create cache."
        )

    except GoogleAPIError as e:
        debug_log("Firestore Google API error", str(e))

        if PLACES_CACHE:
            return PLACES_CACHE

        local_places = load_places_from_local_cache()
        if local_places:
            PLACES_CACHE = local_places
            PLACES_CACHE_TIME = current_time
            return PLACES_CACHE

        raise

    except Exception as e:
        debug_log("Firestore unknown error", str(e))

        if PLACES_CACHE:
            return PLACES_CACHE

        local_places = load_places_from_local_cache()
        if local_places:
            PLACES_CACHE = local_places
            PLACES_CACHE_TIME = current_time
            return PLACES_CACHE

        raise

def save_places_to_local_cache(places: List[Dict[str, Any]]) -> None:
    try:
        payload = {
            "savedAt": now_iso(),
            "count": len(places),
            "places": places,
        }

        with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

        debug_log("Saved places to local cache", {
            "file": LOCAL_CACHE_FILE,
            "count": len(places),
        })

    except Exception as e:
        debug_log("Failed to save local cache", str(e))


def load_places_from_local_cache() -> List[Dict[str, Any]]:
    try:
        if not os.path.exists(LOCAL_CACHE_FILE):
            return []

        with open(LOCAL_CACHE_FILE, "r", encoding="utf-8") as f:
            payload = json.load(f)

        places = payload.get("places", [])

        if not isinstance(places, list):
            return []

        debug_log("Loaded places from local cache", {
            "file": LOCAL_CACHE_FILE,
            "count": len(places),
            "savedAt": payload.get("savedAt"),
        })

        return places

    except Exception as e:
        debug_log("Failed to load local cache", str(e))
        return []

def get_place_by_id(place_id: str) -> Optional[Dict[str, Any]]:
    places = load_places_from_firestore()
    for place in places:
        if place["id"] == place_id:
            return place
    return None


# ============================================================
# SEARCH ENGINE
# ============================================================

def extract_moods(message: str) -> List[str]:
    q = normalize_text(message)
    moods = []

    for mood, words in MOOD_KEYWORDS.items():
        if any(normalize_text(w) in q for w in words):
            moods.append(mood)

    return unique_list(moods)


def extract_districts(message: str) -> List[str]:
    q = normalize_text(message)
    found = []

    for district in KERALA_DISTRICTS:
        if district in q:
            found.append(DISTRICT_ALIASES.get(district, district.title()))

    for alias, canonical in DISTRICT_ALIASES.items():
        if alias in q:
            found.append(canonical)

    return unique_list(found)


def word_overlap_score(query: str, text: str) -> float:
    q_words = set([w for w in normalize_text(query).split() if len(w) >= 3])
    t_words = set([w for w in normalize_text(text).split() if len(w) >= 3])

    if not q_words or not t_words:
        return 0

    overlap = q_words.intersection(t_words)
    return len(overlap) / max(len(q_words), 1)


def place_search_score(query: str, place: Dict[str, Any]) -> float:
    q = normalize_text(query)

    name = normalize_text(place.get("name", ""))
    region = normalize_text(place.get("region", ""))
    distance = normalize_text(place.get("distance", ""))
    description = normalize_text(place.get("description", ""))
    tags = " ".join(normalize_text(t) for t in place.get("tags", []))
    search_blob = place.get("search_blob", "")

    score = 0.0

    if name and name in q:
        score += 120

    if q and q in name:
        score += 80

    if q and q in search_blob:
        score += 25

    query_tokens = [t for t in q.split() if len(t) >= 3]

    for token in query_tokens:
        if token in name:
            score += 24
        if token in region:
            score += 14
        if token in tags:
            score += 12
        if token in distance:
            score += 6
        if token in description:
            score += 4

    moods = extract_moods(query)

    for mood in moods:
        mood_words = MOOD_KEYWORDS.get(mood, [])
        text_blob = f"{name} {region} {tags} {description}"
        if any(normalize_text(w) in text_blob for w in mood_words):
            score += 30

    districts = extract_districts(query)

    for district in districts:
        d = normalize_text(district)
        if d in distance or d in description or d in tags or d in region:
            score += 20

    score += word_overlap_score(query, search_blob) * 20

    rating = safe_float(place.get("rating"), 0)
    reviews = safe_int(place.get("userRatings"), 0)

    score += rating * 1.8
    score += min(reviews / 150, 8)

    return score


def search_places(
    query: str,
    places: List[Dict[str, Any]],
    limit: int = 8,
    min_score: float = 5.0
) -> List[Dict[str, Any]]:
    scored = []

    for place in places:
        score = place_search_score(query, place)

        if score >= min_score:
            p = dict(place)
            p["_score"] = round(score, 2)
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)

    return [p for _, p in scored[:limit]]


def trending_score(place: Dict[str, Any]) -> float:
    rating = safe_float(place.get("rating"), 0)
    reviews = safe_int(place.get("userRatings"), 0)

    return (rating * 20) + (reviews * 1.5)


def get_trending_places(
    places: List[Dict[str, Any]],
    limit: int = 8
) -> List[Dict[str, Any]]:
    return sorted(places, key=trending_score, reverse=True)[:limit]


def find_place_by_name(
    query: str,
    places: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    q = normalize_text(query)

    for place in places:
        name = normalize_text(place.get("name", ""))

        if name and name in q:
            return place

    best = None
    best_score = 0

    for place in places:
        score = place_search_score(query, place)

        if score > best_score:
            best = place
            best_score = score

    if best and best_score >= 45:
        return best

    return None


def find_multiple_places_in_message(
    message: str,
    places: List[Dict[str, Any]],
    limit: int = 4
) -> List[Dict[str, Any]]:
    q = normalize_text(message)
    found = []

    for place in places:
        name = normalize_text(place.get("name", ""))

        if name and name in q:
            found.append(place)

    if len(found) >= 2:
        return found[:limit]

    searched = search_places(message, places, limit=limit, min_score=35)

    result = found[:]

    found_ids = {p["id"] for p in result}

    for place in searched:
        if place["id"] not in found_ids:
            result.append(place)
            found_ids.add(place["id"])

    return result[:limit]


# ============================================================
# INTENT DETECTION
# ============================================================

def contains_any(text: str, phrases: List[str]) -> bool:
    q = normalize_text(text)
    return any(normalize_text(p) in q for p in phrases)


def detect_intent(message: str) -> str:
    q = normalize_text(message)

    if contains_any(q, INTENT_KEYWORDS["travel_time"]) and (
        "from" in q or "to" in q or "distance" in q or "route" in q
    ):
        return "travel_time"

    if contains_any(q, INTENT_KEYWORDS["compare"]):
        places = re.split(r"\s+or\s+|\s+vs\s+|\s+versus\s+", q)
        if len(places) >= 2:
            return "compare"

    if contains_any(q, INTENT_KEYWORDS["trip_plan"]):
        return "trip_plan"

    if contains_any(q, INTENT_KEYWORDS["food"]):
        return "food"

    if contains_any(q, INTENT_KEYWORDS["family_safety"]):
        return "family_safety"

    if contains_any(q, INTENT_KEYWORDS["best_time"]):
        return "best_time"

    if contains_any(q, INTENT_KEYWORDS["recommendation"]):
        return "recommendation"

    if contains_any(q, INTENT_KEYWORDS["details"]):
        return "details"

    return "general"


def extract_day_count(message: str) -> int:
    q = normalize_text(message)

    patterns = [
        r"(\d+)\s*day",
        r"(\d+)\s*days",
    ]

    for pattern in patterns:
        m = re.search(pattern, q)
        if m:
            return max(1, min(10, int(m.group(1))))

    if "weekend" in q:
        return 2

    if "one day" in q or "1 day" in q:
        return 1

    return 2


def extract_origin_destination(
    message: str,
    places: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    q = normalize_text(message)

    origin = None
    destination = None

    # Pattern: from X to Y
    m = re.search(r"from\s+([a-z\s]+?)\s+to\s+([a-z\s]+)", q)
    if m:
        origin = m.group(1).strip()
        destination = m.group(2).strip()

    # Pattern: Y from X
    if not origin:
        m = re.search(r"(.+?)\s+from\s+([a-z\s]+)", q)
        if m:
            before_from = m.group(1).strip()
            after_from = m.group(2).strip()

            origin = after_from

            for place in places:
                name = normalize_text(place.get("name", ""))

                if name and name in before_from:
                    destination = place.get("name")
                    break

            if not destination:
                destination = before_from

    # Pattern: distance to Y from X
    if not origin:
        m = re.search(r"to\s+([a-z\s]+?)\s+from\s+([a-z\s]+)", q)
        if m:
            destination = m.group(1).strip()
            origin = m.group(2).strip()

    if not destination:
        found = find_place_by_name(message, places)
        if found:
            destination = found.get("name")

    if origin:
        origin = clean_location_name(origin)

    if destination:
        destination = clean_location_name(destination)

        known = find_place_by_name(destination, places)
        if known:
            destination = known.get("name")

    return origin, destination


def clean_location_name(location: str) -> str:
    location = normalize_text(location)

    remove_words = [
        "how",
        "much",
        "time",
        "will",
        "take",
        "travel",
        "distance",
        "route",
        "drive",
        "go",
        "reach",
        "by",
        "car",
        "bike",
        "bus",
        "train",
        "from",
        "to",
    ]

    words = [
        w for w in location.split()
        if w not in remove_words
    ]

    cleaned = " ".join(words).strip()

    for district in KERALA_DISTRICTS:
        if district in cleaned:
            return DISTRICT_ALIASES.get(district, district.title())

    return cleaned.title() if cleaned else location.title()


# ============================================================
# MAPS / TRAVEL SERVICE
# ============================================================

def number_to_time_from_distance_km(distance_km: float) -> str:
    if distance_km <= 0:
        return "travel time not available"

    # Kerala road average. Hill routes are slower.
    if distance_km > 120:
        avg_speed_kmph = 38
    else:
        avg_speed_kmph = 42

    hours = distance_km / avg_speed_kmph

    h = int(hours)
    m = int((hours - h) * 60)

    if h <= 0:
        return f"around {m} minutes"

    if m <= 10:
        return f"around {h} hours"

    return f"around {h} hr {m} min"


def make_google_maps_direction_link(origin: str, destination: str) -> str:
    origin_q = requests.utils.quote(f"{origin}, Kerala, India")
    dest_q = requests.utils.quote(f"{destination}, Kerala, India")
    return f"https://www.google.com/maps/dir/?api=1&origin={origin_q}&destination={dest_q}&travelmode=driving"


def get_google_maps_distance(
    origin: str,
    destination: str
) -> Optional[Dict[str, Any]]:
    if not GOOGLE_MAPS_API_KEY:
        return None

    try:
        url = "https://maps.googleapis.com/maps/api/distancematrix/json"

        params = {
            "origins": f"{origin}, Kerala, India",
            "destinations": f"{destination}, Kerala, India",
            "mode": "driving",
            "key": GOOGLE_MAPS_API_KEY,
        }

        res = requests.get(url, params=params, timeout=12)
        data = res.json()

        if data.get("status") != "OK":
            debug_log("Google Maps API status not OK", data)
            return None

        rows = data.get("rows", [])
        if not rows:
            return None

        elements = rows[0].get("elements", [])
        if not elements:
            return None

        element = elements[0]

        if element.get("status") != "OK":
            debug_log("Google Maps element status not OK", element)
            return None

        return {
            "origin": origin,
            "destination": destination,
            "distance_text": element["distance"]["text"],
            "distance_meters": element["distance"]["value"],
            "duration_text": element["duration"]["text"],
            "duration_seconds": element["duration"]["value"],
            "source": "google_maps",
            "maps_url": make_google_maps_direction_link(origin, destination),
        }

    except Exception as e:
        debug_log("Google Maps exception", str(e))
        return None


def get_fallback_travel_estimate(
    origin: str,
    destination: str
) -> Optional[Dict[str, Any]]:
    if not origin or not destination:
        return None

    o = normalize_text(origin)
    d = normalize_text(destination)

    key = (o, d)
    reverse_key = (d, o)

    distance_km = None

    if key in KNOWN_DISTANCES_KM:
        distance_km = KNOWN_DISTANCES_KM[key]
    elif reverse_key in KNOWN_DISTANCES_KM:
        distance_km = KNOWN_DISTANCES_KM[reverse_key]

    if not distance_km:
        return None

    return {
        "origin": origin,
        "destination": destination,
        "distance_text": f"around {distance_km} km",
        "duration_text": number_to_time_from_distance_km(distance_km),
        "source": "estimate",
        "maps_url": make_google_maps_direction_link(origin, destination),
    }


def get_travel_info(
    message: str,
    places: List[Dict[str, Any]]
) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    origin, destination = extract_origin_destination(message, places)

    if not origin or not destination:
        return None, origin, destination

    travel_info = get_google_maps_distance(origin, destination)

    if not travel_info:
        travel_info = get_fallback_travel_estimate(origin, destination)

    return travel_info, origin, destination


# ============================================================
# RESPONSE BUILDERS
# ============================================================

def format_place_short(place: Dict[str, Any]) -> str:
    return (
        f"{place.get('name', 'Unknown')} "
        f"({place.get('region', 'Kerala')}) - "
        f"{safe_float(place.get('rating'), 0):.1f}★, "
        f"{safe_int(place.get('userRatings'), 0)} reviews, "
        f"Best time: {place.get('bestTime', 'Any time')}, "
        f"Distance: {place.get('distance', 'Not available')}. "
        f"Tags: {', '.join(place.get('tags', []))}. "
        f"Description: {clean_for_display(place.get('description', ''), 450)}"
    )


def build_system_prompt() -> str:
    return """
You are Kerala AI Guide, a smart travel assistant for the KeralaTour app.

Core behavior:
- Answer only as a Kerala tourism guide.
- Use KeralaTour database context whenever available.
- If relevant places are provided, prefer them over general knowledge.
- If travel/distance info is provided, use it clearly.
- If Google Maps info is unavailable, say "around" or "approximately".
- Be helpful, warm, and practical.
- Use light Malayalam greeting like "Namaskaram 🙏" sometimes, not in every sentence.
- Do not invent live prices, hotel availability, taxi booking confirmation, or ticket availability.
- Do not say you booked anything.
- If user asks outside Kerala tourism, politely bring back to Kerala travel.

Answer quality:
- Give direct answer first.
- Then short explanation.
- For place questions, include: best time, highlights, who it is good for, and one practical tip.
- For trip plans, give day-wise plan.
- For comparison, give clear recommendation.
- For family/kids, include safety and easy-access tips.
- For monsoon/waterfalls/hills, mention road/rain caution.
- Keep answer compact but useful.

Format:
- 2 to 5 short paragraphs, or bullets if needed.
- Use simple English.
- Avoid long generic text.
"""


def build_ai_context(
    message: str,
    intent: str,
    relevant_places: List[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    moods: List[str],
    districts: List[str],
    day_count: int
) -> str:
    places_context = "\n".join(
        f"- {format_place_short(p)}"
        for p in relevant_places[:MAX_PLACES_CONTEXT]
    )

    travel_context = json.dumps(travel_info, indent=2) if travel_info else ""

    return f"""
User message:
{message}

Detected intent:
{intent}

Detected moods:
{", ".join(moods) if moods else "None"}

Detected districts/locations:
{", ".join(districts) if districts else "None"}

Trip day count if relevant:
{day_count}

Relevant KeralaTour database places:
{places_context if places_context else "No strong database match found."}

Travel/distance info:
{travel_context if travel_context else "No travel info available."}

Now answer as Kerala AI Guide.
"""


def call_groq_ai(
    user_message: str,
    intent: str,
    relevant_places: List[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
    moods: List[str],
    districts: List[str],
    day_count: int
) -> str:
    if not groq_client:
        raise RuntimeError("Groq API key not configured")

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        }
    ]

    for item in history[-6:]:
        role = item.get("role")
        content = safe_text(item.get("content"))

        if role in ["user", "assistant"] and content:
            messages.append({
                "role": role,
                "content": content[:1000],
            })

    messages.append({
        "role": "user",
        "content": build_ai_context(
            message=user_message,
            intent=intent,
            relevant_places=relevant_places,
            travel_info=travel_info,
            moods=moods,
            districts=districts,
            day_count=day_count,
        ),
    })

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.45,
        max_tokens=850,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# LOCAL FALLBACK REPLIES
# ============================================================

def reply_for_travel_time(
    travel_info: Optional[Dict[str, Any]],
    origin: Optional[str],
    destination: Optional[str]
) -> str:
    if travel_info:
        maps_line = ""
        if travel_info.get("maps_url"):
            maps_line = f"\n\nRoute map: {travel_info['maps_url']}"

        source_note = (
            "Google Maps estimate"
            if travel_info.get("source") == "google_maps"
            else "rough Kerala road estimate"
        )

        return (
            f"Namaskaram 🙏 From {travel_info['origin']} to {travel_info['destination']}, "
            f"it is {travel_info['distance_text']} and usually takes "
            f"{travel_info['duration_text']} by road.\n\n"
            f"This is a {source_note}. If it is a hill route, start early morning because traffic, rain, and ghat roads can slow the trip."
            f"{maps_line}"
        )

    return (
        f"Namaskaram 🙏 I understood you are asking about travel time"
        f"{f' from {origin}' if origin else ''}"
        f"{f' to {destination}' if destination else ''}. "
        f"I could not get exact route data now. Try asking like: "
        f"'How much time from Malappuram to Munnar?'"
    )


def reply_for_recommendation(
    places: List[Dict[str, Any]],
    relevant_places: List[Dict[str, Any]]
) -> str:
    top = relevant_places or get_trending_places(places, 6)

    if not top:
        return "Namaskaram 🙏 I don’t have enough places loaded from the database yet."

    lines = []

    for i, place in enumerate(top[:6], start=1):
        lines.append(
            f"{i}. {place['name']} — {place['region']}, "
            f"{safe_float(place['rating']):.1f}★, best time: {place['bestTime']}"
        )

    return (
        "Here are good Kerala places from your KeralaTour database 🌴\n\n"
        + "\n".join(lines)
        + "\n\nTell me your starting city and number of days, and I can make a route plan."
    )


def reply_for_place_details(place: Dict[str, Any]) -> str:
    return (
        f"Namaskaram 🙏 {place['name']} is a {place['region']} destination in Kerala.\n\n"
        f"Rating: {safe_float(place['rating']):.1f}★ with {safe_int(place['userRatings'])} reviews\n"
        f"Best time: {place['bestTime']}\n"
        f"Distance: {place['distance'] or 'Not available'}\n\n"
        f"{clean_for_display(place['description'], 550)}"
    )


def reply_for_compare(places: List[Dict[str, Any]]) -> str:
    if len(places) < 2:
        return "Tell me two places to compare, like: 'Munnar or Wayanad which is better?'"

    a = places[0]
    b = places[1]

    a_score = trending_score(a)
    b_score = trending_score(b)

    winner = a if a_score >= b_score else b

    return (
        f"Good question 🌴\n\n"
        f"{a['name']}: {a['region']}, {safe_float(a['rating']):.1f}★, best time {a['bestTime']}.\n"
        f"{b['name']}: {b['region']}, {safe_float(b['rating']):.1f}★, best time {b['bestTime']}.\n\n"
        f"My suggestion: choose {winner['name']} if you want the safer overall pick from your database rating/review strength.\n\n"
        f"Simple rule: choose hill stations for cool climate and views, backwaters for slow peaceful travel, beaches for sunsets, and wildlife places for adventure."
    )


def reply_for_trip_plan(
    message: str,
    places: List[Dict[str, Any]],
    relevant_places: List[Dict[str, Any]],
    day_count: int
) -> str:
    selected = relevant_places or get_trending_places(places, 6)

    if not selected:
        return "I need places loaded in the database to create a trip plan."

    days = []

    for day in range(1, day_count + 1):
        place = selected[(day - 1) % len(selected)]
        days.append(
            f"Day {day}: {place['name']} — explore {place['region']}. "
            f"Best time: {place['bestTime']}. Tip: start early and keep the plan light."
        )

    return (
        f"Here is a simple {day_count}-day Kerala trip plan based on your database 🌴\n\n"
        + "\n".join(days)
        + "\n\nShare your starting place, budget, and travel style, and I can make it more accurate."
    )


def local_fallback_reply(
    message: str,
    intent: str,
    places: List[Dict[str, Any]],
    relevant_places: List[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    origin: Optional[str],
    destination: Optional[str],
    day_count: int
) -> str:
    if intent == "travel_time":
        return reply_for_travel_time(travel_info, origin, destination)

    if intent == "compare":
        compared = find_multiple_places_in_message(message, places, limit=3)
        return reply_for_compare(compared)

    if intent == "trip_plan":
        return reply_for_trip_plan(message, places, relevant_places, day_count)

    if intent == "recommendation":
        return reply_for_recommendation(places, relevant_places)

    if intent == "best_time":
        if relevant_places:
            p = relevant_places[0]
            return (
                f"For {p['name']}, the best time to visit is {p['bestTime']}.\n\n"
                f"{clean_for_display(p['description'], 400)}"
            )

        return (
            "For most Kerala trips, October to March is the most comfortable season. "
            "For monsoon beauty, June to September is beautiful, but waterfalls and hill roads need extra care."
        )

    if relevant_places:
        return reply_for_place_details(relevant_places[0])

    return (
        "Namaskaram 🙏 I can help you with Kerala destinations, travel time, trip plans, family-safe places, beaches, backwaters, hills, wildlife, best time to visit, and route ideas."
    )


# ============================================================
# CHAT ORCHESTRATION
# ============================================================

def build_chat_result(
    message: str,
    history: List[Dict[str, str]]
) -> Dict[str, Any]:
    places = load_places_from_firestore()

    intent = detect_intent(message)
    moods = extract_moods(message)
    districts = extract_districts(message)
    day_count = extract_day_count(message)

    relevant_places = search_places(
        query=message,
        places=places,
        limit=MAX_PLACES_CONTEXT,
        min_score=5,
    )

    if intent in ["recommendation", "general"] and not relevant_places:
        relevant_places = get_trending_places(places, limit=MAX_PLACES_CONTEXT)

    if intent == "compare":
        relevant_places = find_multiple_places_in_message(
            message,
            places,
            limit=MAX_PLACES_CONTEXT,
        )

    if intent == "trip_plan" and not relevant_places:
        relevant_places = get_trending_places(places, limit=MAX_PLACES_CONTEXT)

    travel_info = None
    origin = None
    destination = None

    if intent == "travel_time":
        travel_info, origin, destination = get_travel_info(message, places)

        if destination:
            destination_place = find_place_by_name(destination, places)
            if destination_place:
                existing_ids = {p["id"] for p in relevant_places}
                if destination_place["id"] not in existing_ids:
                    relevant_places.insert(0, destination_place)

    ai_used = False

    try:
        reply = call_groq_ai(
            user_message=message,
            intent=intent,
            relevant_places=relevant_places,
            travel_info=travel_info,
            history=history if isinstance(history, list) else [],
            moods=moods,
            districts=districts,
            day_count=day_count,
        )
        ai_used = True

    except Exception as ai_error:
        debug_log("AI error, using fallback", str(ai_error))

        reply = local_fallback_reply(
            message=message,
            intent=intent,
            places=places,
            relevant_places=relevant_places,
            travel_info=travel_info,
            origin=origin,
            destination=destination,
            day_count=day_count,
        )

    return {
        "reply": reply,
        "intent": intent,
        "aiUsed": ai_used,
        "moods": moods,
        "districts": districts,
        "dayCount": day_count,
        "origin": origin,
        "destination": destination,
        "matchedPlaces": [
            {
                "id": p["id"],
                "name": p["name"],
                "region": p["region"],
                "rating": p["rating"],
                "userRatings": p["userRatings"],
                "bestTime": p["bestTime"],
                "distance": p["distance"],
                "imageUrl": p["imageUrl"],
                "tags": p["tags"],
                "score": p.get("_score"),
            }
            for p in relevant_places[:6]
        ],
        "travelInfo": travel_info,
        "placeCount": len(places),
        "timestamp": now_iso(),
    }


# ============================================================
# API ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "name": APP_NAME,
        "groqConfigured": bool(GROQ_API_KEY),
        "googleMapsConfigured": bool(GOOGLE_MAPS_API_KEY),
        "routes": [
            "/api/chat",
            "/api/places",
            "/api/search?q=munnar",
            "/api/trending",
            "/api/place/<id>",
            "/api/cache/refresh",
        ],
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(force=True) or {}

        message = safe_text(body.get("message"))
        history = body.get("history", [])

        if not message:
            return jsonify({
                "reply": "Please ask me something about Kerala travel.",
                "error": "empty_message",
            }), 400

        result = build_chat_result(message, history)
        return jsonify(result)

    except Exception as e:
        debug_log("Server error in /api/chat", {
            "error": str(e),
            "trace": traceback.format_exc(),
        })

        return jsonify({
            "reply": "Sorry, I had a small server issue. Please try again 🙏",
            "error": str(e),
        }), 500


@app.route("/api/places", methods=["GET"])
def places_api():
    try:
        limit = safe_int(request.args.get("limit"), 100)
        limit = max(1, min(limit, 2000))

        places = load_places_from_firestore()

        return jsonify({
            "count": len(places),
            "places": places[:limit],
            "cacheAgeSeconds": round(time.time() - PLACES_CACHE_TIME, 2),
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
        }), 500


@app.route("/api/search", methods=["GET"])
def search_api():
    try:
        q = safe_text(request.args.get("q"))
        limit = safe_int(request.args.get("limit"), 10)
        limit = max(1, min(limit, 50))

        if not q:
            return jsonify({
                "query": q,
                "count": 0,
                "places": [],
            })

        places = load_places_from_firestore()
        results = search_places(q, places, limit=limit, min_score=3)

        return jsonify({
            "query": q,
            "count": len(results),
            "places": results,
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
        }), 500


@app.route("/api/trending", methods=["GET"])
def trending_api():
    try:
        limit = safe_int(request.args.get("limit"), 20)
        limit = max(1, min(limit, 100))

        places = load_places_from_firestore()
        results = get_trending_places(places, limit=limit)

        return jsonify({
            "count": len(results),
            "places": results,
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
        }), 500


@app.route("/api/place/<place_id>", methods=["GET"])
def place_detail_api(place_id: str):
    try:
        place = get_place_by_id(place_id)

        if not place:
            return jsonify({
                "error": "place_not_found",
                "placeId": place_id,
            }), 404

        return jsonify({
            "place": place,
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
        }), 500


@app.route("/api/cache/refresh", methods=["POST", "GET"])
def refresh_cache_api():
    try:
        places = load_places_from_firestore(force=True)

        return jsonify({
            "status": "ok",
            "count": len(places),
            "cacheTime": PLACES_CACHE_TIME,
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
        }), 500


# ============================================================
# DEV HELPERS
# ============================================================

@app.route("/api/debug/intent", methods=["POST"])
def debug_intent_api():
    body = request.get_json(force=True) or {}
    message = safe_text(body.get("message"))

    places = load_places_from_firestore()

    travel_info, origin, destination = get_travel_info(message, places)

    return jsonify({
        "message": message,
        "intent": detect_intent(message),
        "moods": extract_moods(message),
        "districts": extract_districts(message),
        "dayCount": extract_day_count(message),
        "origin": origin,
        "destination": destination,
        "travelInfo": travel_info,
        "matches": search_places(message, places, limit=5, min_score=3),
    })


#####################################################
#. import version 2
#======================================
# from v2 import v2_bp

# app.register_blueprint(v2_bp)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(APP_NAME)
    print("=" * 60)
    print(f"Groq configured: {bool(GROQ_API_KEY)}")
    print(f"Google Maps configured: {bool(GOOGLE_MAPS_API_KEY)}")
    print(f"Firebase service account: {FIREBASE_SERVICE_ACCOUNT}")
    print(f"Model: {GROQ_MODEL}")
    print(f"Port: {PORT}")
    print("=" * 60)

    debug_mode = os.getenv("ENV", "development") == "development"
    app.run(host="0.0.0.0", port=PORT, debug=debug_mode)