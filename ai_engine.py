import os
import re
import json
import time
import math
import hashlib
import traceback
import requests
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from functools import wraps
from collections import defaultdict

from flask import Flask, request, jsonify, g
from flask_cors import CORS
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, firestore
from google.api_core.exceptions import ResourceExhausted, GoogleAPIError
from groq import Groq


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")
PORT                = int(os.getenv("PORT", "5000"))
FIREBASE_SA         = os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json")
GROQ_MODEL          = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant")

MAX_PLACES_CONTEXT  = 12
CACHE_TTL_SECONDS   = 60 * 60 * 6   # 6 hours
LOCAL_CACHE_FILE    = "places_cache.json"
CONV_CACHE_TTL      = 60 * 60 * 2   # 2 hours conversation memory
MAX_HISTORY_TURNS   = 10
RATE_LIMIT_PER_MIN  = 30
MAX_TOKENS_CHAT     = 1200
MAX_TOKENS_PLAN     = 2000

APP_NAME = "KeralaTour AI Backend v2"


# ============================================================
# FLASK INIT
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# FIREBASE INIT
# ============================================================

if not firebase_admin._apps:
    cred = credentials.Certificate(FIREBASE_SA)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ============================================================
# GROQ INIT
# ============================================================

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ============================================================
# GLOBAL CACHES
# ============================================================

PLACES_CACHE:      List[Dict[str, Any]] = []
PLACES_CACHE_TIME: float = 0
PLACES_CACHE_LOCK  = threading.Lock()

# Conversation memory: session_id -> list of {role, content, ts}
CONV_MEMORY: Dict[str, List[Dict]] = {}
CONV_LOCK   = threading.Lock()

# Response cache (hash of context -> reply, ts)
RESPONSE_CACHE: Dict[str, Dict] = {}
RESPONSE_CACHE_TTL = 60 * 30   # 30 min

# Rate limiting: ip -> [timestamps]
RATE_LIMIT_STORE: Dict[str, List[float]] = defaultdict(list)


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
    text = text.replace("–", "-").replace("—", "-").replace("'", "").replace("'", "")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_for_display(text: str, limit: int = 600) -> str:
    text = safe_text(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > limit:
        return text[:limit].rstrip() + "..."
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

def hash_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def debug_log(label: str, data: Any = None):
    print(f"[{now_iso()}] {label}")
    if data is not None:
        try:
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2000])
        except Exception:
            print(str(data)[:2000])


# ============================================================
# RATE LIMITING
# ============================================================

def check_rate_limit(ip: str) -> bool:
    """Returns True if request is allowed."""
    now = time.time()
    window_start = now - 60
    RATE_LIMIT_STORE[ip] = [t for t in RATE_LIMIT_STORE[ip] if t > window_start]
    if len(RATE_LIMIT_STORE[ip]) >= RATE_LIMIT_PER_MIN:
        return False
    RATE_LIMIT_STORE[ip].append(now)
    return True

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.remote_addr or "unknown"
        if not check_rate_limit(ip):
            return jsonify({"error": "Rate limit exceeded. Please wait a moment."}), 429
        return f(*args, **kwargs)
    return decorated


# ============================================================
# CONVERSATION MEMORY
# ============================================================

def get_session_history(session_id: str) -> List[Dict]:
    if not session_id:
        return []
    with CONV_LOCK:
        history = CONV_MEMORY.get(session_id, [])
        # Prune expired turns
        cutoff = time.time() - CONV_CACHE_TTL
        history = [h for h in history if h.get("ts", 0) > cutoff]
        CONV_MEMORY[session_id] = history
        return list(history)

def save_session_turn(session_id: str, role: str, content: str):
    if not session_id:
        return
    with CONV_LOCK:
        if session_id not in CONV_MEMORY:
            CONV_MEMORY[session_id] = []
        CONV_MEMORY[session_id].append({
            "role": role,
            "content": content[:1200],
            "ts": time.time(),
        })
        # Keep only last N turns
        if len(CONV_MEMORY[session_id]) > MAX_HISTORY_TURNS * 2:
            CONV_MEMORY[session_id] = CONV_MEMORY[session_id][-(MAX_HISTORY_TURNS * 2):]

def clear_session(session_id: str):
    with CONV_LOCK:
        CONV_MEMORY.pop(session_id, None)

def prune_old_sessions():
    """Remove stale sessions - call periodically."""
    cutoff = time.time() - CONV_CACHE_TTL
    with CONV_LOCK:
        to_delete = [
            sid for sid, turns in CONV_MEMORY.items()
            if not turns or turns[-1].get("ts", 0) < cutoff
        ]
        for sid in to_delete:
            del CONV_MEMORY[sid]


# ============================================================
# DOMAIN DATA
# ============================================================

KERALA_DISTRICTS = [
    "thiruvananthapuram", "trivandrum", "kollam", "pathanamthitta",
    "alappuzha", "alleppey", "kottayam", "idukki", "ernakulam",
    "kochi", "thrissur", "palakkad", "malappuram", "kozhikode",
    "calicut", "wayanad", "kannur", "kasaragod",
]

DISTRICT_ALIASES = {
    "trivandrum":  "Thiruvananthapuram",
    "tvm":         "Thiruvananthapuram",
    "alleppey":    "Alappuzha",
    "calicut":     "Kozhikode",
    "cochin":      "Kochi",
    "ernakulam":   "Kochi",
}

MOOD_KEYWORDS = {
    "romantic":   ["romantic", "couple", "honeymoon", "peaceful", "sunset", "private", "calm", "intimate"],
    "family":     ["family", "kids", "children", "safe", "picnic", "parents", "baby", "elder", "grandparents"],
    "adventure":  ["adventure", "trek", "trekking", "hiking", "forest", "wildlife", "camping", "offroad", "rappelling", "zipline"],
    "budget":     ["budget", "cheap", "low cost", "affordable", "free", "less money", "economical", "backpacker"],
    "luxury":     ["luxury", "premium", "resort", "spa", "five star", "5 star", "exclusive", "boutique"],
    "solo":       ["solo", "alone", "single", "solo travel", "by myself", "backpacker"],
    "monsoon":    ["rain", "monsoon", "rainy", "waterfall", "green", "lush"],
    "beach":      ["beach", "coastal", "sea", "sunset", "shore", "lighthouse", "sand"],
    "hill":       ["hill", "hill station", "munnar", "vagamon", "ponmudi", "tea", "misty", "cool", "mountain"],
    "backwater":  ["backwater", "lake", "boating", "houseboat", "canal", "kayaking", "punting"],
    "wildlife":   ["wildlife", "forest", "sanctuary", "national park", "elephant", "tiger", "bird", "safari"],
    "heritage":   ["heritage", "temple", "fort", "palace", "museum", "church", "mosque", "culture", "history"],
    "food":       ["food", "eat", "restaurant", "dish", "seafood", "sadya", "biriyani", "local food", "cuisine"],
    "photography":["photo", "photography", "instagram", "scenic", "sunrise", "viewpoint", "landscape"],
    "spiritual":  ["temple", "pilgrimage", "spiritual", "meditation", "yoga", "ashram", "prayer", "divine"],
}

INTENT_KEYWORDS = {
    "travel_time":   ["how much time", "how long", "time take", "travel time", "distance", "km", "from", "reach", "route", "drive", "bus", "train", "car", "bike", "scooter", "how far"],
    "recommendation":["best", "top", "trending", "popular", "highest rated", "suggest", "recommend", "where to go", "places", "must visit", "hidden gem", "underrated"],
    "trip_plan":     ["plan", "trip", "itinerary", "2 day", "3 day", "4 day", "5 day", "one day", "1 day", "weekend", "schedule", "route plan", "travel plan", "7 day", "week"],
    "best_time":     ["when", "best time", "season", "month", "monsoon", "summer", "winter", "visit time", "which month", "when to visit"],
    "food":          ["food", "eat", "restaurant", "local food", "dish", "breakfast", "lunch", "dinner", "where to eat", "must try", "street food"],
    "family_safety": ["safe", "family", "kids", "child", "children", "parents", "elder", "baby", "wheelchair", "accessible"],
    "budget_tips":   ["budget", "cheap", "how much", "cost", "price", "affordable", "money", "spend", "expenses", "how expensive"],
    "accommodation": ["stay", "hotel", "resort", "homestay", "guesthouse", "houseboat", "hostel", "where to stay", "accommodation", "book"],
    "compare":       ["compare", "which is better", "better", "or", "vs", "versus", "difference between", "which one"],
    "details":       ["what about", "tell me about", "details", "information", "info", "describe", "explain"],
    "weather":       ["weather", "climate", "temperature", "rain", "fog", "cold", "hot", "humid"],
    "transport":     ["how to reach", "nearest airport", "bus stand", "railway", "transport", "taxi", "auto", "ferry", "boat"],
    "offbeat":       ["offbeat", "hidden", "secret", "unexplored", "less crowded", "peaceful", "unknown", "avoid crowd"],
    "permits":       ["permit", "entry fee", "ticket", "forest permit", "pass", "booking"],
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
    ("kannur", "munnar"): 310,
    ("malappuram", "wayanad"): 120,
    ("kochi", "wayanad"): 270,
    ("ernakulam", "wayanad"): 270,
    ("calicut", "wayanad"): 90,
    ("kozhikode", "wayanad"): 90,
    ("thrissur", "wayanad"): 230,
    ("kannur", "wayanad"): 130,
    ("kochi", "alleppey"): 55,
    ("ernakulam", "alleppey"): 55,
    ("malappuram", "alleppey"): 190,
    ("kochi", "alappuzha"): 55,
    ("malappuram", "alappuzha"): 190,
    ("trivandrum", "alappuzha"): 155,
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
    ("kochi", "kovalam"): 225,
    ("trivandrum", "kovalam"): 16,
    ("kochi", "kollam"): 70,
    ("trivandrum", "kollam"): 70,
    ("kochi", "thrissur"): 76,
    ("kochi", "palakkad"): 150,
    ("kochi", "kozhikode"): 220,
    ("kochi", "kannur"): 295,
    ("kochi", "kasaragod"): 380,
    ("kozhikode", "kannur"): 92,
    ("kozhikode", "kasaragod"): 175,
    ("thrissur", "palakkad"): 75,
    ("trivandrum", "kanyakumari"): 87,
    ("kochi", "kanyakumari"): 300,
    ("kochi", "guruvayur"): 80,
    ("thrissur", "guruvayur"): 29,
    ("malappuram", "kozhikode"): 55,
    ("malappuram", "thrissur"): 110,
    ("malappuram", "palakkad"): 100,
    ("malappuram", "kannur"): 145,
    ("trivandrum", "kumarakom"): 170,
    ("kochi", "kumarakom"): 50,
    ("kochi", "periyar"): 190,
    ("trivandrum", "periyar"): 230,
}

SEASON_GUIDE = {
    "oct_mar": {
        "label": "October to March",
        "mood": "Best season — pleasant weather, clear skies, ideal for all places.",
        "highlights": ["beaches", "backwaters", "hill stations", "heritage sites"],
    },
    "apr_may": {
        "label": "April to May",
        "mood": "Hot and humid. Good for waterfalls and hill stations. Avoid low-altitude beaches midday.",
        "highlights": ["hill stations", "Munnar", "Vagamon", "Ponmudi"],
    },
    "jun_sep": {
        "label": "June to September",
        "mood": "Monsoon — lush greenery, roaring waterfalls, fewer crowds. Hill roads need caution.",
        "highlights": ["waterfalls", "Athirappilly", "Wayanad forests", "tea estates"],
    },
}

KERALA_FOOD_GUIDE = {
    "must_try": [
        "Kerala Sadya (banana leaf feast)",
        "Karimeen Pollichathu (pearl spot fish in banana leaf)",
        "Appam with Stew",
        "Malabar Biryani",
        "Prawn Moilee",
        "Kerala Porotta with beef curry",
        "Puttu and Kadala curry",
        "Banana chips",
        "Pazham Pori (banana fritters)",
        "Coconut fish curry",
    ],
    "by_region": {
        "Malabar (North Kerala)": ["Malabar Biryani", "Thalassery Biryani", "Pathiri", "Erachi curry"],
        "Central Kerala": ["Karimeen Pollichathu", "Kuttanadan duck curry", "Tapioca with fish"],
        "South Kerala": ["Nadan Chicken curry", "Fish Moilee", "Avial", "Sadya"],
    },
}

TRANSPORT_INFO = {
    "airports": {
        "kochi": "Cochin International Airport (COK) — largest, best connectivity",
        "trivandrum": "Thiruvananthapuram International Airport (TRV)",
        "kozhikode": "Calicut International Airport (CCJ) — serves North Kerala",
        "kannur": "Kannur International Airport (CNN) — new, serves far north",
    },
    "trains": "Kerala has excellent rail connectivity along the coast. Major junctions: Thiruvananthapuram, Ernakulam, Thrissur, Kozhikode.",
    "buses": "KSRTC operates state-wide. Private AC sleepers available for long routes.",
    "local": "Auto-rickshaws, local buses, Ola/Uber in cities. Ferry boats in backwaters.",
}

BUDGET_GUIDE = {
    "backpacker": {
        "label": "Budget / Backpacker",
        "daily_inr": "800–1500 INR/day",
        "stay": "Hostels, budget guesthouses",
        "food": "Local hotels, mess",
        "transport": "KSRTC buses, shared autos",
    },
    "mid": {
        "label": "Mid-range",
        "daily_inr": "2000–5000 INR/day",
        "stay": "3-star hotels, homestays",
        "food": "Restaurants + local meals",
        "transport": "Private cabs, some trains",
    },
    "luxury": {
        "label": "Luxury",
        "daily_inr": "8000+ INR/day",
        "stay": "Heritage resorts, luxury houseboats, 5-star",
        "food": "Resort dining, specialty restaurants",
        "transport": "Private car, chartered ferry",
    },
}


# ============================================================
# FIRESTORE SERVICE
# ============================================================

def normalize_place(doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    description = safe_text(
        data.get("ai_description") or data.get("description"),
        "A beautiful destination in Kerala."
    )
    name   = safe_text(data.get("name"), "Unknown Place")
    region = safe_text(data.get("region"), "Kerala")
    tags   = split_tags(data.get("tags"))

    search_blob = " ".join([
        doc_id, name, region,
        safe_text(data.get("distance")),
        safe_text(data.get("bestTime")),
        description,
        " ".join(tags),
    ])

    return {
        "id":            doc_id,
        "name":          name,
        "region":        region,
        "distance":      safe_text(data.get("distance"), ""),
        "imageUrl":      safe_text(data.get("imageUrl"), ""),
        "rating":        safe_float(data.get("rating"), 0.0),
        "userRatings":   safe_int(data.get("userRatings"), 0),
        "ratingTotal":   safe_float(data.get("ratingTotal"), 0.0),
        "bestTime":      safe_text(data.get("bestTime"), "Any time"),
        "description":   description,
        "raw_description": safe_text(data.get("description"), ""),
        "ai_description":  safe_text(data.get("ai_description"), ""),
        "tags":          tags,
        "source":        safe_text(data.get("source"), ""),
        "entryFee":      safe_text(data.get("entryFee"), ""),
        "openHours":     safe_text(data.get("openHours"), ""),
        "category":      safe_text(data.get("category"), ""),
        "lat":           safe_float(data.get("lat"), 0.0),
        "lng":           safe_float(data.get("lng"), 0.0),
        "search_blob":   normalize_text(search_blob),
    }


def load_places_from_firestore(force: bool = False) -> List[Dict[str, Any]]:
    global PLACES_CACHE, PLACES_CACHE_TIME

    with PLACES_CACHE_LOCK:
        current_time = time.time()
        cache_valid = (
            PLACES_CACHE
            and not force
            and (current_time - PLACES_CACHE_TIME) < CACHE_TTL_SECONDS
        )
        if cache_valid:
            return PLACES_CACHE

        if not PLACES_CACHE and not force:
            local = load_places_from_local_cache()
            if local:
                PLACES_CACHE = local
                PLACES_CACHE_TIME = current_time
                debug_log("Using local cache", {"count": len(PLACES_CACHE)})
                return PLACES_CACHE

        try:
            places = []
            for doc in db.collection("places").stream():
                data = doc.to_dict() or {}
                places.append(normalize_place(doc.id, data))
            PLACES_CACHE = places
            PLACES_CACHE_TIME = current_time
            save_places_to_local_cache(PLACES_CACHE)
            debug_log("Firestore refreshed", {"count": len(PLACES_CACHE)})
            return PLACES_CACHE

        except ResourceExhausted as e:
            debug_log("Firestore quota exceeded", str(e))
            if PLACES_CACHE:
                return PLACES_CACHE
            local = load_places_from_local_cache()
            if local:
                PLACES_CACHE = local
                PLACES_CACHE_TIME = current_time
                return PLACES_CACHE
            raise RuntimeError("Firestore quota exceeded. No local cache available.")

        except Exception as e:
            debug_log("Firestore error", str(e))
            if PLACES_CACHE:
                return PLACES_CACHE
            local = load_places_from_local_cache()
            if local:
                PLACES_CACHE = local
                PLACES_CACHE_TIME = current_time
                return PLACES_CACHE
            raise


def save_places_to_local_cache(places: List[Dict[str, Any]]):
    try:
        with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump({"savedAt": now_iso(), "count": len(places), "places": places}, f, ensure_ascii=False)
        debug_log("Local cache saved", {"count": len(places)})
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
        debug_log("Local cache loaded", {"count": len(places), "savedAt": payload.get("savedAt")})
        return places
    except Exception as e:
        debug_log("Failed to load local cache", str(e))
        return []


def get_place_by_id(place_id: str) -> Optional[Dict[str, Any]]:
    places = load_places_from_firestore()
    for p in places:
        if p["id"] == place_id:
            return p
    return None


def upsert_place(place_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Write or update a place in Firestore and refresh cache."""
    data["updatedAt"] = now_iso()
    db.collection("places").document(place_id).set(data, merge=True)
    load_places_from_firestore(force=True)
    return get_place_by_id(place_id) or {}


# ============================================================
# SEARCH ENGINE (enhanced)
# ============================================================

def extract_moods(message: str) -> List[str]:
    q = normalize_text(message)
    return unique_list([
        mood for mood, words in MOOD_KEYWORDS.items()
        if any(normalize_text(w) in q for w in words)
    ])


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
    q_words = {w for w in normalize_text(query).split() if len(w) >= 3}
    t_words = {w for w in normalize_text(text).split() if len(w) >= 3}
    if not q_words or not t_words:
        return 0.0
    return len(q_words & t_words) / max(len(q_words), 1)


def place_search_score(query: str, place: Dict[str, Any]) -> float:
    q     = normalize_text(query)
    name  = normalize_text(place.get("name", ""))
    region= normalize_text(place.get("region", ""))
    dist  = normalize_text(place.get("distance", ""))
    desc  = normalize_text(place.get("description", ""))
    tags  = " ".join(normalize_text(t) for t in place.get("tags", []))
    cat   = normalize_text(place.get("category", ""))
    blob  = place.get("search_blob", "")

    score = 0.0

    if name and name in q:
        score += 120
    if q and q in name:
        score += 80
    if q and q in blob:
        score += 25

    tokens = [t for t in q.split() if len(t) >= 3]
    for token in tokens:
        if token in name:   score += 24
        if token in region: score += 14
        if token in tags:   score += 12
        if token in cat:    score += 10
        if token in dist:   score += 6
        if token in desc:   score += 4

    moods = extract_moods(query)
    blob_text = f"{name} {region} {tags} {desc}"
    for mood in moods:
        mood_words = MOOD_KEYWORDS.get(mood, [])
        if any(normalize_text(w) in blob_text for w in mood_words):
            score += 30

    districts = extract_districts(query)
    for district in districts:
        d = normalize_text(district)
        if any(d in x for x in [dist, desc, tags, region]):
            score += 20

    score += word_overlap_score(query, blob) * 20

    rating  = safe_float(place.get("rating"), 0)
    reviews = safe_int(place.get("userRatings"), 0)
    score  += rating * 1.8
    score  += min(reviews / 150, 8)

    return score


def search_places(
    query: str,
    places: List[Dict[str, Any]],
    limit: int = 8,
    min_score: float = 5.0,
    category_filter: Optional[str] = None,
    mood_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    scored = []
    for place in places:
        if category_filter and normalize_text(place.get("category", "")) != normalize_text(category_filter):
            continue
        if mood_filter and mood_filter not in " ".join(place.get("tags", [])).lower():
            continue
        score = place_search_score(query, place)
        if score >= min_score:
            p = dict(place)
            p["_score"] = round(score, 2)
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:limit]]


def trending_score(place: Dict[str, Any]) -> float:
    rating  = safe_float(place.get("rating"), 0)
    reviews = safe_int(place.get("userRatings"), 0)
    return (rating * 20) + (reviews * 1.5)


def get_trending_places(places: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    return sorted(places, key=trending_score, reverse=True)[:limit]


def find_place_by_name(query: str, places: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    q = normalize_text(query)
    for place in places:
        name = normalize_text(place.get("name", ""))
        if name and name in q:
            return place

    best, best_score = None, 0
    for place in places:
        score = place_search_score(query, place)
        if score > best_score:
            best, best_score = place, score

    return best if best and best_score >= 45 else None


def find_multiple_places_in_message(message: str, places: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    q = normalize_text(message)
    found = [p for p in places if normalize_text(p.get("name", "")) and normalize_text(p.get("name", "")) in q]
    if len(found) >= 2:
        return found[:limit]
    searched = search_places(message, places, limit=limit, min_score=35)
    found_ids = {p["id"] for p in found}
    result = found[:]
    for p in searched:
        if p["id"] not in found_ids:
            result.append(p)
            found_ids.add(p["id"])
    return result[:limit]


def get_similar_places(place: Dict[str, Any], places: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    """Find places similar to a given place based on tags and region."""
    tags_query = " ".join(place.get("tags", []))
    region_query = place.get("region", "")
    query = f"{tags_query} {region_query}"
    results = search_places(query, places, limit=limit + 1, min_score=10)
    return [p for p in results if p["id"] != place["id"]][:limit]


# ============================================================
# INTENT DETECTION (enhanced)
# ============================================================

def contains_any(text: str, phrases: List[str]) -> bool:
    q = normalize_text(text)
    return any(normalize_text(p) in q for p in phrases)


def detect_intent(message: str) -> str:
    q = normalize_text(message)

    if contains_any(q, INTENT_KEYWORDS["travel_time"]) and any(w in q for w in ["from", "to", "distance", "route", "far", "reach"]):
        return "travel_time"
    if contains_any(q, INTENT_KEYWORDS["compare"]):
        parts = re.split(r"\s+or\s+|\s+vs\s+|\s+versus\s+|\s+better\s+", q)
        if len(parts) >= 2:
            return "compare"
    if contains_any(q, INTENT_KEYWORDS["trip_plan"]):
        return "trip_plan"
    if contains_any(q, INTENT_KEYWORDS["accommodation"]):
        return "accommodation"
    if contains_any(q, INTENT_KEYWORDS["budget_tips"]):
        return "budget_tips"
    if contains_any(q, INTENT_KEYWORDS["food"]):
        return "food"
    if contains_any(q, INTENT_KEYWORDS["transport"]):
        return "transport"
    if contains_any(q, INTENT_KEYWORDS["weather"]):
        return "weather"
    if contains_any(q, INTENT_KEYWORDS["family_safety"]):
        return "family_safety"
    if contains_any(q, INTENT_KEYWORDS["best_time"]):
        return "best_time"
    if contains_any(q, INTENT_KEYWORDS["offbeat"]):
        return "offbeat"
    if contains_any(q, INTENT_KEYWORDS["permits"]):
        return "permits"
    if contains_any(q, INTENT_KEYWORDS["recommendation"]):
        return "recommendation"
    if contains_any(q, INTENT_KEYWORDS["details"]):
        return "details"
    return "general"


def extract_day_count(message: str) -> int:
    q = normalize_text(message)
    for pattern in [r"(\d+)\s*-?\s*day", r"(\d+)\s*days?"]:
        m = re.search(pattern, q)
        if m:
            return max(1, min(14, int(m.group(1))))
    if "weekend" in q: return 2
    if "one day" in q or "1 day" in q: return 1
    if "week" in q: return 7
    return 3


def extract_origin_destination(message: str, places: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    q = normalize_text(message)
    origin = destination = None

    m = re.search(r"from\s+([a-z\s]+?)\s+to\s+([a-z\s]+)", q)
    if m:
        origin = m.group(1).strip()
        destination = m.group(2).strip()

    if not origin:
        m = re.search(r"to\s+([a-z\s]+?)\s+from\s+([a-z\s]+)", q)
        if m:
            destination = m.group(1).strip()
            origin = m.group(2).strip()

    if not origin:
        m = re.search(r"(.+?)\s+from\s+([a-z\s]+)", q)
        if m:
            origin = m.group(2).strip()
            before = m.group(1).strip()
            found = find_place_by_name(before, places)
            destination = found.get("name") if found else before

    if not destination:
        found = find_place_by_name(message, places)
        if found:
            destination = found.get("name")

    if origin:      origin      = clean_location_name(origin)
    if destination:
        destination = clean_location_name(destination)
        known = find_place_by_name(destination, places)
        if known:
            destination = known.get("name")

    return origin, destination


def clean_location_name(location: str) -> str:
    location = normalize_text(location)
    remove = {"how", "much", "time", "will", "take", "travel", "distance", "route",
               "drive", "go", "reach", "by", "car", "bike", "bus", "train", "from",
               "to", "long", "far", "reach", "get"}
    words = [w for w in location.split() if w not in remove]
    cleaned = " ".join(words).strip()
    for district in KERALA_DISTRICTS:
        if district in cleaned:
            return DISTRICT_ALIASES.get(district, district.title())
    return cleaned.title() if cleaned else location.title()


def extract_budget_level(message: str) -> Optional[str]:
    q = normalize_text(message)
    if any(w in q for w in ["budget", "cheap", "affordable", "low cost", "backpacker"]):
        return "backpacker"
    if any(w in q for w in ["luxury", "premium", "5 star", "resort"]):
        return "luxury"
    return "mid"


# ============================================================
# MAPS / TRAVEL SERVICE
# ============================================================

def estimate_time_from_km(distance_km: float) -> str:
    if distance_km <= 0:
        return "travel time not available"
    avg_speed = 38 if distance_km > 120 else 42
    hours = distance_km / avg_speed
    h, m = int(hours), int((hours - int(hours)) * 60)
    if h <= 0: return f"around {m} minutes"
    if m <= 10: return f"around {h} hours"
    return f"around {h} hr {m} min"


def make_maps_link(origin: str, destination: str) -> str:
    oq = requests.utils.quote(f"{origin}, Kerala, India")
    dq = requests.utils.quote(f"{destination}, Kerala, India")
    return f"https://www.google.com/maps/dir/?api=1&origin={oq}&destination={dq}&travelmode=driving"


def get_google_maps_distance(origin: str, destination: str) -> Optional[Dict[str, Any]]:
    if not GOOGLE_MAPS_API_KEY:
        return None
    try:
        res = requests.get(
            "https://maps.googleapis.com/maps/api/distancematrix/json",
            params={
                "origins": f"{origin}, Kerala, India",
                "destinations": f"{destination}, Kerala, India",
                "mode": "driving",
                "key": GOOGLE_MAPS_API_KEY,
            },
            timeout=12,
        )
        data = res.json()
        if data.get("status") != "OK": return None
        element = data["rows"][0]["elements"][0]
        if element.get("status") != "OK": return None
        return {
            "origin": origin, "destination": destination,
            "distance_text": element["distance"]["text"],
            "distance_meters": element["distance"]["value"],
            "duration_text": element["duration"]["text"],
            "duration_seconds": element["duration"]["value"],
            "source": "google_maps",
            "maps_url": make_maps_link(origin, destination),
        }
    except Exception as e:
        debug_log("Google Maps exception", str(e))
        return None


def get_fallback_travel_estimate(origin: str, destination: str) -> Optional[Dict[str, Any]]:
    if not origin or not destination:
        return None
    o, d = normalize_text(origin), normalize_text(destination)
    km = KNOWN_DISTANCES_KM.get((o, d)) or KNOWN_DISTANCES_KM.get((d, o))
    if not km:
        return None
    return {
        "origin": origin, "destination": destination,
        "distance_text": f"around {km} km",
        "duration_text": estimate_time_from_km(km),
        "source": "estimate",
        "maps_url": make_maps_link(origin, destination),
    }


def get_travel_info(message: str, places: List[Dict[str, Any]]) -> Tuple[Optional[Dict], Optional[str], Optional[str]]:
    origin, destination = extract_origin_destination(message, places)
    if not origin or not destination:
        return None, origin, destination
    travel_info = get_google_maps_distance(origin, destination) or get_fallback_travel_estimate(origin, destination)
    return travel_info, origin, destination


# ============================================================
# AI PROMPT BUILDERS (greatly enhanced)
# ============================================================

def build_system_prompt(intent: str) -> str:
    base = """You are Kerala AI Guide — an expert, warm, and deeply knowledgeable Kerala tourism assistant inside the KeralaTour app.

Core identity:
- You are a human-feeling guide, not a robotic chatbot.
- You love Kerala and speak about it with genuine enthusiasm.
- You have deep local knowledge: places, food, transport, culture, seasons, hidden gems.
- You speak in warm, conversational English. Occasionally use Malayalam words naturally (Namaskaram, Onam, Sadhya, Vallamkali, etc.).

Strict rules:
- ONLY answer Kerala tourism questions. Politely redirect all others.
- NEVER invent hotel prices, live availability, or real-time ticket data.
- NEVER claim to have booked anything.
- NEVER hallucinate non-existent places or events.
- Always use the database context provided if relevant places are included.
- If travel data is from Google Maps, say so. If estimated, say "approximately".

Answer quality standards:
- Direct answer FIRST, then supporting detail.
- For place questions: name → location → what makes it special → best time → who it suits → one insider tip.
- For trip plans: structured day-by-day with timing, must-sees, food recommendations, and practical tips.
- For food: regional context + best place to eat it.
- For comparisons: clear recommendation with reasoning, not both-sides hedging.
- For monsoon/hill/trekking: always include safety note.
- For families: highlight kid-friendly features and safety.
- For budget travelers: share money-saving tricks specific to Kerala.

Response format:
- Conversational paragraphs for most answers. Use bullet lists sparingly and only when listing multiple discrete items.
- Keep responses rich but scannable. Max 4-5 short paragraphs, or a structured plan if asked.
- End with an actionable follow-up offer (e.g., "Want me to plan a full day at Munnar?").
"""

    intent_addons = {
        "trip_plan": "\n\nFor trip plans: Create a detailed day-by-day itinerary with morning/afternoon/evening breakdown. Include: places to visit with timing, local food to try, distance/travel info between stops, accommodation type suggestion, and one unique local experience per day.",
        "food": "\n\nFor food questions: Name the dish, its regional origin, what makes it special, best places to find it (type of eatery), and the best time to eat it (breakfast/lunch/dinner).",
        "budget_tips": "\n\nFor budget advice: Give concrete INR estimates where possible. Tip on free attractions, budget transport, local eating vs tourist restaurants, and when to travel for cheaper rates.",
        "accommodation": "\n\nFor accommodation: Describe the types available (budget/mid/luxury), what's unique about Kerala stays (houseboats, tree houses, heritage homes), approximate price range, and booking tips.",
        "transport": "\n\nFor transport questions: Cover all realistic options (bus, train, taxi, ferry, auto) with rough costs and time. Mention which is most scenic or recommended.",
        "weather": "\n\nFor weather: Give month-by-month guidance. Mention rainfall, temperature range, what's open vs closed, and specific clothing/gear tips.",
    }

    return base + intent_addons.get(intent, "")


def format_place_for_ai(place: Dict[str, Any]) -> str:
    entry_fee = f", Entry: {place['entryFee']}" if place.get("entryFee") else ""
    hours = f", Hours: {place['openHours']}" if place.get("openHours") else ""
    return (
        f"[{place.get('name')}] Region: {place.get('region')} | "
        f"Rating: {safe_float(place.get('rating')):.1f}★ ({safe_int(place.get('userRatings'))} reviews) | "
        f"Best time: {place.get('bestTime')} | Distance: {place.get('distance', 'N/A')}"
        f"{entry_fee}{hours} | "
        f"Tags: {', '.join(place.get('tags', []))} | "
        f"Description: {clean_for_display(place.get('description', ''), 500)}"
    )


def build_ai_context(
    message: str,
    intent: str,
    relevant_places: List[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    moods: List[str],
    districts: List[str],
    day_count: int,
    budget_level: Optional[str],
    history_summary: str,
) -> str:
    places_context = "\n".join(
        f"- {format_place_for_ai(p)}"
        for p in relevant_places[:MAX_PLACES_CONTEXT]
    )

    travel_context = json.dumps(travel_info, indent=2) if travel_info else "Not applicable"

    seasonal_hint = ""
    current_month = datetime.utcnow().month
    if current_month in [10, 11, 12, 1, 2, 3]:
        seasonal_hint = "Current season: Peak (Oct-Mar) — ideal for most Kerala travel."
    elif current_month in [4, 5]:
        seasonal_hint = "Current season: Pre-monsoon (Apr-May) — hot, best for hills."
    else:
        seasonal_hint = "Current season: Monsoon (Jun-Sep) — lush, waterfalls peak, hill roads caution."

    return f"""
## User Message
{message}

## Detected Context
- Intent: {intent}
- Moods/vibes: {', '.join(moods) if moods else 'Not specified'}
- Districts mentioned: {', '.join(districts) if districts else 'None'}
- Trip duration: {day_count} day(s)
- Budget level: {budget_level or 'Not specified'}
- {seasonal_hint}

## Conversation Memory (Recent)
{history_summary or 'No prior context.'}

## Relevant Places from KeralaTour Database
{places_context if places_context else 'No strong database match — use general Kerala knowledge.'}

## Travel/Distance Info
{travel_context}

---
Now respond as Kerala AI Guide. Be specific, warm, and genuinely helpful.
"""


def summarize_history(history: List[Dict]) -> str:
    """Create a compact summary of conversation history for the AI context."""
    if not history:
        return ""
    lines = []
    for item in history[-6:]:
        role = item.get("role", "")
        content = safe_text(item.get("content", ""))[:300]
        if role == "user":
            lines.append(f"User asked: {content}")
        elif role == "assistant":
            lines.append(f"You replied: {content}")
    return "\n".join(lines)


# ============================================================
# GROQ AI CALLER (with retry + fallback model)
# ============================================================

def call_groq_ai(
    user_message: str,
    intent: str,
    relevant_places: List[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
    moods: List[str],
    districts: List[str],
    day_count: int,
    budget_level: Optional[str] = None,
    max_tokens: int = MAX_TOKENS_CHAT,
    use_fallback_model: bool = False,
) -> str:
    if not groq_client:
        raise RuntimeError("Groq API key not configured")

    model = GROQ_FALLBACK_MODEL if use_fallback_model else GROQ_MODEL
    history_summary = summarize_history(history)

    messages = [
        {"role": "system", "content": build_system_prompt(intent)},
    ]

    # Include last few turns for real multi-turn context
    for item in history[-MAX_HISTORY_TURNS:]:
        role = item.get("role")
        content = safe_text(item.get("content"))
        if role in ["user", "assistant"] and content:
            messages.append({"role": role, "content": content[:1000]})

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
            budget_level=budget_level,
            history_summary=history_summary,
        ),
    })

    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.55,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        if not use_fallback_model:
            debug_log("Primary model failed, trying fallback", str(e))
            return call_groq_ai(
                user_message=user_message,
                intent=intent,
                relevant_places=relevant_places,
                travel_info=travel_info,
                history=history,
                moods=moods,
                districts=districts,
                day_count=day_count,
                budget_level=budget_level,
                max_tokens=max_tokens,
                use_fallback_model=True,
            )
        raise


# ============================================================
# RESPONSE CACHE
# ============================================================

def get_cached_response(cache_key: str) -> Optional[str]:
    entry = RESPONSE_CACHE.get(cache_key)
    if entry and (time.time() - entry["ts"]) < RESPONSE_CACHE_TTL:
        return entry["reply"]
    return None


def set_cached_response(cache_key: str, reply: str):
    RESPONSE_CACHE[cache_key] = {"reply": reply, "ts": time.time()}
    # Prune old entries if cache is getting large
    if len(RESPONSE_CACHE) > 500:
        cutoff = time.time() - RESPONSE_CACHE_TTL
        expired = [k for k, v in RESPONSE_CACHE.items() if v["ts"] < cutoff]
        for k in expired:
            del RESPONSE_CACHE[k]


# ============================================================
# LOCAL FALLBACK REPLIES (enhanced)
# ============================================================

def reply_for_travel_time(travel_info, origin, destination) -> str:
    if travel_info:
        src = "Google Maps" if travel_info.get("source") == "google_maps" else "road estimate"
        maps_line = f"\n\n📍 Route: {travel_info['maps_url']}" if travel_info.get("maps_url") else ""
        return (
            f"Namaskaram 🙏 From {travel_info['origin']} to {travel_info['destination']}: "
            f"distance is {travel_info['distance_text']}, travel time is roughly {travel_info['duration_text']} by road.\n\n"
            f"This is a {src}. If it involves ghat roads (like to Munnar or Wayanad), "
            f"start early morning — fog, rain, and traffic can slow things considerably."
            f"{maps_line}"
        )
    return (
        f"I understood your travel time query"
        f"{f' from {origin}' if origin else ''}"
        f"{f' to {destination}' if destination else ''}."
        f" Try phrasing like: 'How long from Kochi to Munnar?' for best results."
    )


def reply_for_recommendation(places, relevant_places) -> str:
    top = relevant_places or get_trending_places(places, 6)
    if not top:
        return "Database is loading. Please try again in a moment."
    lines = [
        f"{i}. **{p['name']}** — {p['region']}, {safe_float(p['rating']):.1f}★, best time: {p['bestTime']}"
        for i, p in enumerate(top[:6], 1)
    ]
    return (
        "Here are top Kerala destinations from the KeralaTour database 🌴\n\n"
        + "\n".join(lines)
        + "\n\nTell me your starting city, trip duration, and travel style — I'll plan a full route."
    )


def reply_for_compare(places) -> str:
    if len(places) < 2:
        return "Tell me two places to compare, e.g. 'Munnar or Wayanad — which is better?'"
    a, b = places[0], places[1]
    winner = a if trending_score(a) >= trending_score(b) else b
    return (
        f"Great question! Here's a quick comparison 🌴\n\n"
        f"**{a['name']}**: {a['region']} — {safe_float(a['rating']):.1f}★, best visited {a['bestTime']}.\n"
        f"**{b['name']}**: {b['region']} — {safe_float(b['rating']):.1f}★, best visited {b['bestTime']}.\n\n"
        f"My pick: **{winner['name']}** edges ahead on database ratings. "
        f"Quick rule: choose hill stations for cool air and views, "
        f"backwaters for slow peaceful travel, beaches for sunset vibes."
    )


def reply_for_trip_plan(message, places, relevant_places, day_count) -> str:
    selected = relevant_places or get_trending_places(places, day_count + 2)
    if not selected:
        return "I need the places database to create a trip plan. Please try again."
    days = []
    for day in range(1, day_count + 1):
        place = selected[(day - 1) % len(selected)]
        days.append(
            f"**Day {day}**: {place['name']} ({place['region']}) — "
            f"Best time to visit: {place['bestTime']}. "
            f"Start early, explore the main highlights, and try local food in the evening."
        )
    return (
        f"Here's a {day_count}-day Kerala trip outline 🌴\n\n"
        + "\n".join(days)
        + "\n\nShare your starting city, budget, and travel style for a detailed personalized plan."
    )


def reply_for_budget(budget_level: str) -> str:
    guide = BUDGET_GUIDE.get(budget_level, BUDGET_GUIDE["mid"])
    return (
        f"Kerala travel budget guide ({guide['label']}) 💰\n\n"
        f"Daily estimate: {guide['daily_inr']}\n"
        f"Stay: {guide['stay']}\n"
        f"Food: {guide['food']}\n"
        f"Transport: {guide['transport']}\n\n"
        f"Kerala tip: Off-peak (June–September) is cheapest. "
        f"Local KSRTC buses save a lot. Beach shacks and small local restaurants give the best value food."
    )


def reply_for_food(message: str) -> str:
    must_try = KERALA_FOOD_GUIDE["must_try"][:6]
    by_region = KERALA_FOOD_GUIDE["by_region"]
    region_lines = "\n".join([f"• **{r}**: {', '.join(dishes)}" for r, dishes in by_region.items()])
    return (
        "Kerala food is a culinary journey 🍽️\n\n"
        f"Must-try dishes: {', '.join(must_try)}\n\n"
        f"By region:\n{region_lines}\n\n"
        "For the most authentic experience, try a proper Kerala Sadya (banana leaf feast) on a Sunday or festival day."
    )


def reply_for_transport(message: str) -> str:
    airports = TRANSPORT_INFO["airports"]
    airport_lines = "\n".join([f"• **{k.title()}**: {v}" for k, v in airports.items()])
    return (
        "How to get around Kerala 🚂\n\n"
        f"**Airports:**\n{airport_lines}\n\n"
        f"**Trains:** {TRANSPORT_INFO['trains']}\n\n"
        f"**Buses:** {TRANSPORT_INFO['buses']}\n\n"
        f"**Local:** {TRANSPORT_INFO['local']}"
    )


def local_fallback_reply(
    message: str,
    intent: str,
    places: List[Dict[str, Any]],
    relevant_places: List[Dict[str, Any]],
    travel_info: Optional[Dict],
    origin: Optional[str],
    destination: Optional[str],
    day_count: int,
    budget_level: Optional[str],
) -> str:
    if intent == "travel_time":
        return reply_for_travel_time(travel_info, origin, destination)
    if intent == "compare":
        return reply_for_compare(find_multiple_places_in_message(message, places, limit=3))
    if intent == "trip_plan":
        return reply_for_trip_plan(message, places, relevant_places, day_count)
    if intent == "recommendation":
        return reply_for_recommendation(places, relevant_places)
    if intent == "food":
        return reply_for_food(message)
    if intent == "budget_tips":
        return reply_for_budget(budget_level or "mid")
    if intent == "transport":
        return reply_for_transport(message)
    if intent == "best_time":
        if relevant_places:
            p = relevant_places[0]
            return (
                f"For **{p['name']}**, the best time to visit is **{p['bestTime']}**.\n\n"
                f"{clean_for_display(p['description'], 400)}"
            )
        return (
            "Best time guide for Kerala 📅\n\n"
            "**October–March**: Peak season — pleasant weather, perfect for all places.\n"
            "**April–May**: Hot but great for hill stations like Munnar, Vagamon.\n"
            "**June–September**: Monsoon — lush, waterfalls roar, but hill roads need caution.\n\n"
            "For most travelers, November to February is the sweet spot."
        )
    if relevant_places:
        p = relevant_places[0]
        return (
            f"**{p['name']}** ({p['region']}) — {safe_float(p['rating']):.1f}★\n\n"
            f"Best time: {p['bestTime']}\n"
            f"Distance: {p['distance'] or 'Not listed'}\n\n"
            f"{clean_for_display(p['description'], 550)}"
        )
    return (
        "Namaskaram 🙏 I can help with Kerala destinations, travel time, trip plans, "
        "food guide, transport info, best seasons, family-safe places, budget tips, and hidden gems. "
        "What would you like to know?"
    )


# ============================================================
# CHAT ORCHESTRATION (fully enhanced)
# ============================================================

def build_chat_result(
    message: str,
    history: List[Dict[str, str]],
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    places = load_places_from_firestore()

    intent       = detect_intent(message)
    moods        = extract_moods(message)
    districts    = extract_districts(message)
    day_count    = extract_day_count(message)
    budget_level = extract_budget_level(message)

    # Merge session memory with request history
    if session_id:
        mem_history = get_session_history(session_id)
        # Deduplicate: use memory history + any new items in request history
        merged = mem_history[:]
        seen_contents = {h.get("content", "") for h in merged}
        for h in history:
            if h.get("content", "") not in seen_contents:
                merged.append(h)
                seen_contents.add(h.get("content", ""))
        history = merged

    # Build search query — enrich with moods/districts
    search_query = message
    if moods:
        search_query += " " + " ".join(moods)
    if districts:
        search_query += " " + " ".join(districts)

    relevant_places = search_places(
        query=search_query,
        places=places,
        limit=MAX_PLACES_CONTEXT,
        min_score=5,
    )

    if intent in ["recommendation", "general"] and not relevant_places:
        relevant_places = get_trending_places(places, limit=MAX_PLACES_CONTEXT)

    if intent == "compare":
        relevant_places = find_multiple_places_in_message(message, places, limit=MAX_PLACES_CONTEXT)

    if intent == "trip_plan" and not relevant_places:
        relevant_places = get_trending_places(places, limit=MAX_PLACES_CONTEXT)

    if intent == "offbeat":
        # Sort by least reviews (less popular = more offbeat)
        relevant_places = sorted(places, key=lambda p: safe_int(p.get("userRatings"), 0))[:MAX_PLACES_CONTEXT]

    travel_info = origin = destination = None
    if intent == "travel_time":
        travel_info, origin, destination = get_travel_info(message, places)
        if destination:
            dest_place = find_place_by_name(destination, places)
            if dest_place:
                ids = {p["id"] for p in relevant_places}
                if dest_place["id"] not in ids:
                    relevant_places.insert(0, dest_place)

    max_tokens = MAX_TOKENS_PLAN if intent == "trip_plan" else MAX_TOKENS_CHAT

    # Check response cache (skip for trip plans — always fresh)
    cache_key = None
    if intent not in ["trip_plan"] and not session_id:
        cache_key = hash_str(f"{intent}:{normalize_text(message)}:{len(relevant_places)}")
        cached = get_cached_response(cache_key)
        if cached:
            debug_log("Response cache hit", {"intent": intent})
            return _wrap_result(
                reply=cached,
                intent=intent,
                ai_used=True,
                moods=moods,
                districts=districts,
                day_count=day_count,
                origin=origin,
                destination=destination,
                relevant_places=relevant_places,
                travel_info=travel_info,
                places=places,
                cached=True,
            )

    ai_used = False
    reply = ""

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
            budget_level=budget_level,
            max_tokens=max_tokens,
        )
        ai_used = True

        # Cache successful AI responses
        if cache_key:
            set_cached_response(cache_key, reply)

    except Exception as ai_err:
        debug_log("AI error → fallback", str(ai_err))
        reply = local_fallback_reply(
            message=message,
            intent=intent,
            places=places,
            relevant_places=relevant_places,
            travel_info=travel_info,
            origin=origin,
            destination=destination,
            day_count=day_count,
            budget_level=budget_level,
        )

    # Save to session memory
    if session_id:
        save_session_turn(session_id, "user", message)
        save_session_turn(session_id, "assistant", reply)

    return _wrap_result(
        reply=reply,
        intent=intent,
        ai_used=ai_used,
        moods=moods,
        districts=districts,
        day_count=day_count,
        origin=origin,
        destination=destination,
        relevant_places=relevant_places,
        travel_info=travel_info,
        places=places,
        cached=False,
    )


def _wrap_result(
    reply, intent, ai_used, moods, districts, day_count,
    origin, destination, relevant_places, travel_info, places, cached=False,
) -> Dict[str, Any]:
    return {
        "reply":         reply,
        "intent":        intent,
        "aiUsed":        ai_used,
        "cached":        cached,
        "moods":         moods,
        "districts":     districts,
        "dayCount":      day_count,
        "origin":        origin,
        "destination":   destination,
        "matchedPlaces": [
            {
                "id":          p["id"],
                "name":        p["name"],
                "region":      p["region"],
                "rating":      p["rating"],
                "userRatings": p["userRatings"],
                "bestTime":    p["bestTime"],
                "distance":    p["distance"],
                "imageUrl":    p["imageUrl"],
                "tags":        p["tags"],
                "category":    p.get("category", ""),
                "entryFee":    p.get("entryFee", ""),
                "lat":         p.get("lat", 0.0),
                "lng":         p.get("lng", 0.0),
                "score":       p.get("_score"),
            }
            for p in relevant_places[:8]
        ],
        "travelInfo":    travel_info,
        "placeCount":    len(places),
        "timestamp":     now_iso(),
    }


# ============================================================
# API ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    places_loaded = len(PLACES_CACHE)
    sessions_active = len(CONV_MEMORY)
    cache_entries = len(RESPONSE_CACHE)
    return jsonify({
        "status":       "ok",
        "name":         APP_NAME,
        "version":      "2.0.0",
        "groqConfigured": bool(GROQ_API_KEY),
        "googleMapsConfigured": bool(GOOGLE_MAPS_API_KEY),
        "placesLoaded": places_loaded,
        "activeSessions": sessions_active,
        "responseCacheSize": cache_entries,
        "routes": [
            "POST /api/chat",
            "GET  /api/places",
            "GET  /api/search?q=munnar",
            "GET  /api/trending",
            "GET  /api/place/<id>",
            "GET  /api/similar/<id>",
            "POST /api/place/<id>",
            "GET  /api/session/<id>/history",
            "DELETE /api/session/<id>",
            "GET  /api/stats",
            "GET  /api/food-guide",
            "GET  /api/transport",
            "GET  /api/seasons",
            "GET  /api/budget-guide",
            "POST /api/cache/refresh",
            "POST /api/debug/intent",
        ],
    })


@app.route("/api/chat", methods=["POST"])
@rate_limit
def chat():
    try:
        body    = request.get_json(force=True) or {}
        message = safe_text(body.get("message"))
        history = body.get("history", [])
        session_id = safe_text(body.get("sessionId", ""))

        if not message:
            return jsonify({"reply": "Please ask me something about Kerala travel.", "error": "empty_message"}), 400

        if len(message) > 2000:
            return jsonify({"reply": "Message too long. Please keep it under 2000 characters.", "error": "message_too_long"}), 400

        result = build_chat_result(message, history, session_id=session_id or None)
        return jsonify(result)

    except Exception as e:
        debug_log("Server error in /api/chat", {"error": str(e), "trace": traceback.format_exc()})
        return jsonify({
            "reply": "Sorry, I had a small server issue. Please try again 🙏",
            "error": str(e),
        }), 500


@app.route("/api/places", methods=["GET"])
def places_api():
    try:
        limit    = max(1, min(safe_int(request.args.get("limit"), 100), 2000))
        category = safe_text(request.args.get("category", ""))
        mood     = safe_text(request.args.get("mood", ""))

        places = load_places_from_firestore()

        if category:
            places = [p for p in places if normalize_text(p.get("category", "")) == normalize_text(category)]
        if mood and mood in MOOD_KEYWORDS:
            mood_words = MOOD_KEYWORDS[mood]
            places = [
                p for p in places
                if any(normalize_text(w) in p.get("search_blob", "") for w in mood_words)
            ]

        return jsonify({
            "count":            len(places),
            "places":           places[:limit],
            "cacheAgeSeconds":  round(time.time() - PLACES_CACHE_TIME, 2),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def search_api():
    try:
        q        = safe_text(request.args.get("q"))
        limit    = max(1, min(safe_int(request.args.get("limit"), 10), 50))
        category = safe_text(request.args.get("category", ""))
        mood     = safe_text(request.args.get("mood", ""))

        if not q:
            return jsonify({"query": q, "count": 0, "places": []})

        places  = load_places_from_firestore()
        results = search_places(q, places, limit=limit, min_score=3,
                                category_filter=category or None, mood_filter=mood or None)
        return jsonify({"query": q, "count": len(results), "places": results,
                        "moods": extract_moods(q), "districts": extract_districts(q)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trending", methods=["GET"])
def trending_api():
    try:
        limit  = max(1, min(safe_int(request.args.get("limit"), 20), 100))
        places = load_places_from_firestore()
        return jsonify({"count": limit, "places": get_trending_places(places, limit=limit)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/place/<place_id>", methods=["GET"])
def place_detail_api(place_id: str):
    try:
        place = get_place_by_id(place_id)
        if not place:
            return jsonify({"error": "place_not_found", "placeId": place_id}), 404
        return jsonify({"place": place})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/place/<place_id>", methods=["POST"])
def update_place_api(place_id: str):
    """Update or create a place in Firestore."""
    try:
        data = request.get_json(force=True) or {}
        if not data:
            return jsonify({"error": "No data provided"}), 400
        updated = upsert_place(place_id, data)
        return jsonify({"status": "ok", "place": updated})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/similar/<place_id>", methods=["GET"])
def similar_places_api(place_id: str):
    try:
        limit = max(1, min(safe_int(request.args.get("limit"), 4), 10))
        place = get_place_by_id(place_id)
        if not place:
            return jsonify({"error": "place_not_found"}), 404
        places  = load_places_from_firestore()
        similar = get_similar_places(place, places, limit=limit)
        return jsonify({"placeId": place_id, "count": len(similar), "similar": similar})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>/history", methods=["GET"])
def session_history_api(session_id: str):
    try:
        history = get_session_history(session_id)
        return jsonify({"sessionId": session_id, "count": len(history), "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/session/<session_id>", methods=["DELETE"])
def clear_session_api(session_id: str):
    try:
        clear_session(session_id)
        return jsonify({"status": "ok", "sessionId": session_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/stats", methods=["GET"])
def stats_api():
    try:
        places = load_places_from_firestore()
        regions    = {}
        categories = {}
        for p in places:
            r = p.get("region", "Unknown")
            c = p.get("category", "Unknown")
            regions[r]    = regions.get(r, 0) + 1
            categories[c] = categories.get(c, 0) + 1
        avg_rating  = sum(safe_float(p.get("rating")) for p in places) / max(len(places), 1)
        total_reviews = sum(safe_int(p.get("userRatings")) for p in places)
        return jsonify({
            "totalPlaces":     len(places),
            "averageRating":   round(avg_rating, 2),
            "totalReviews":    total_reviews,
            "byRegion":        regions,
            "byCategory":      categories,
            "activeSessions":  len(CONV_MEMORY),
            "responseCacheSize": len(RESPONSE_CACHE),
            "cacheAgeSeconds": round(time.time() - PLACES_CACHE_TIME, 2),
            "timestamp":       now_iso(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/food-guide", methods=["GET"])
def food_guide_api():
    return jsonify(KERALA_FOOD_GUIDE)


@app.route("/api/transport", methods=["GET"])
def transport_api():
    return jsonify(TRANSPORT_INFO)


@app.route("/api/seasons", methods=["GET"])
def seasons_api():
    return jsonify(SEASON_GUIDE)


@app.route("/api/budget-guide", methods=["GET"])
def budget_guide_api():
    level = safe_text(request.args.get("level", ""))
    if level and level in BUDGET_GUIDE:
        return jsonify(BUDGET_GUIDE[level])
    return jsonify(BUDGET_GUIDE)


@app.route("/api/cache/refresh", methods=["POST", "GET"])
def refresh_cache_api():
    try:
        places = load_places_from_firestore(force=True)
        return jsonify({"status": "ok", "count": len(places), "cacheTime": PLACES_CACHE_TIME})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/intent", methods=["POST"])
def debug_intent_api():
    body    = request.get_json(force=True) or {}
    message = safe_text(body.get("message"))
    places  = load_places_from_firestore()
    travel_info, origin, destination = get_travel_info(message, places)
    return jsonify({
        "message":     message,
        "intent":      detect_intent(message),
        "moods":       extract_moods(message),
        "districts":   extract_districts(message),
        "dayCount":    extract_day_count(message),
        "budgetLevel": extract_budget_level(message),
        "origin":      origin,
        "destination": destination,
        "travelInfo":  travel_info,
        "matches":     search_places(message, places, limit=5, min_score=3),
    })


# ============================================================
# BACKGROUND MAINTENANCE
# ============================================================

def _background_maintenance():
    """Runs every 30 minutes to prune stale data."""
    while True:
        time.sleep(60 * 30)
        prune_old_sessions()
        debug_log("Background maintenance: old sessions pruned", {"activeSessions": len(CONV_MEMORY)})


maintenance_thread = threading.Thread(target=_background_maintenance, daemon=True)
maintenance_thread.start()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(APP_NAME)
    print("=" * 60)
    print(f"Groq primary model:   {GROQ_MODEL}")
    print(f"Groq fallback model:  {GROQ_FALLBACK_MODEL}")
    print(f"Google Maps:          {bool(GOOGLE_MAPS_API_KEY)}")
    print(f"Firebase SA:          {FIREBASE_SA}")
    print(f"Port:                 {PORT}")
    print(f"Rate limit:           {RATE_LIMIT_PER_MIN} req/min")
    print("=" * 60)
    app.run(host="0.0.0.0", port=PORT, debug=True)