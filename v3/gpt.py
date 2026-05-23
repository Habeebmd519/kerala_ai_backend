import os
import re
import json
import time
import math
import hashlib
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
# KERALA AI BACKEND V3 - PRACTICAL UI VERSION
# ============================================================
# What this version fixes:
# - Does NOT show image every time.
# - Shows images for list/recommendation requests where cards are useful.
# - Shows one big image only when user explicitly asks image/photo/pic.
# - Supports follow-ups like: "pic one", "explain it", "best time", "how to go".
# - Uses currentPlaceId from frontend to remember selected place.
# - Uses history as backup if frontend forgets currentPlaceId.
# - Avoids repeating imageUrl in normal explanation replies.
# - Gives frontend clear ui instructions: showImages, showBigImage, showCards.
# ============================================================


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

APP_NAME = os.getenv("APP_NAME", "KeralaTour AI Backend V3")
PORT = int(os.getenv("PORT", "5000"))
ENV = os.getenv("ENV", "development")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.45"))
GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "900"))

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "")

FIREBASE_SERVICE_ACCOUNT = os.getenv("FIREBASE_SERVICE_ACCOUNT", "serviceAccountKey.json")
FIREBASE_SERVICE_ACCOUNT_JSON = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON", "")
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "places")

MAX_PLACES_CONTEXT = int(os.getenv("MAX_PLACES_CONTEXT", "10"))
MAX_MATCHED_PLACES_RESPONSE = int(os.getenv("MAX_MATCHED_PLACES_RESPONSE", "8"))
DEFAULT_CARD_LIMIT = int(os.getenv("DEFAULT_CARD_LIMIT", "7"))
EXPLAIN_CARD_LIMIT = int(os.getenv("EXPLAIN_CARD_LIMIT", "1"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", str(60 * 60 * 6)))
LOCAL_CACHE_FILE = os.getenv("LOCAL_CACHE_FILE", "places_cache.json")
REQUEST_TIMEOUT_SECONDS = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "12"))

ENABLE_DEBUG_LOGS = os.getenv("ENABLE_DEBUG_LOGS", "true").lower() in ["1", "true", "yes", "on"]
ENABLE_GROQ = os.getenv("ENABLE_GROQ", "true").lower() in ["1", "true", "yes", "on"]


# ============================================================
# FLASK INIT
# ============================================================

app = Flask(__name__)
CORS(app)


# ============================================================
# FIREBASE INIT
# ============================================================

if not firebase_admin._apps:
    if FIREBASE_SERVICE_ACCOUNT_JSON:
        cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_JSON)
        cred = credentials.Certificate(cred_dict)
    else:
        cred = credentials.Certificate(FIREBASE_SERVICE_ACCOUNT)

    firebase_admin.initialize_app(cred)

db = firestore.client()

NTES_URL = "https://enquiry.indianrail.gov.in/ntes/"

STATION_ALIASES = {
    # major city aliases
    "kochi": "ERS",
    "ernakulam": "ERS",
    "ernakulam junction": "ERS",
    "ernakulam south": "ERS",
    "ernakulam town": "ERN",
    "ernakulam north": "ERN",

    "trivandrum": "TVC",
    "thiruvananthapuram": "TVC",
    "thiruvananthapuram central": "TVC",

    "kollam": "QLN",
    "quilon": "QLN",

    "alappuzha": "ALLP",
    "alleppey": "ALLP",

    "kottayam": "KTYM",

    "thrissur": "TCR",
    "trichur": "TCR",

    "palakkad": "PGT",
    "palghat": "PGT",

    "shoranur": "SRR",

    "kozhikode": "CLT",
    "calicut": "CLT",

    "kannur": "CAN",
    "kasaragod": "KGQ",

    "tirur": "TIR",
    "guruvayur": "GUV",
    "aluva": "AWY",

    # tourist places without railway station
    "munnar": "munnar",
    "wayanad": "wayanad",
    "vagamon": "vagamon",
    "thekkady": "thekkady",
    "bekal": "bekal",
    "athirappilly": "athirappilly",
}


# ============================================================
# GROQ INIT
# ============================================================

groq_client = None

if GROQ_API_KEY and ENABLE_GROQ:
    groq_client = Groq(api_key=GROQ_API_KEY)


# ============================================================
# GLOBAL CACHE
# ============================================================

PLACES_CACHE: List[Dict[str, Any]] = []
PLACES_CACHE_TIME: float = 0
PLACES_INDEX: Dict[str, Any] = {}


# ============================================================
# BASIC UTILS
# ============================================================

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return str(value).strip()
    
def is_near_me_query(message: str) -> bool:
    q = normalize_text(message)
    near_words = [
        "near me",
        "nearby",
        "around me",
        "my location",
        "current location",
        "close to me",
        "nearest",
        "near me for",
    ]
    return any(w in q for w in near_words)


def build_user_location_context(
    user_lat: Optional[float],
    user_lng: Optional[float],
    user_location_text: str = ""
) -> str:
    if user_lat and user_lng:
        return f"{user_lat},{user_lng}"

    if user_location_text:
        return user_location_text

    return ""

def safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return fallback
        return float(value)
    except Exception:
        return fallback


def safe_int(value: Any, fallback: int = 0) -> int:
    try:
        if value is None or value == "":
            return fallback
        return int(float(value))
    except Exception:
        return fallback


def normalize_text(text: Any) -> str:
    text = safe_text(text).lower()
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compact_text(text: Any) -> str:
    return re.sub(r"\s+", " ", safe_text(text)).strip()


def clean_for_display(text: Any, limit: int = 700) -> str:
    text = compact_text(text)
    if len(text) > limit:
        return text[:limit].strip() + "..."
    return text


def slugify(text: str) -> str:
    text = normalize_text(text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def split_tags(tags: Any) -> List[str]:
    if isinstance(tags, list):
        return [safe_text(t) for t in tags if safe_text(t)]

    if isinstance(tags, str):
        raw = tags.strip()
        if raw.startswith("["):
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [safe_text(t) for t in parsed if safe_text(t)]
            except Exception:
                pass
        return [t.strip() for t in raw.split(",") if t.strip()]

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


def debug_log(label: str, data: Any = None) -> None:
    if not ENABLE_DEBUG_LOGS:
        return

    print(f"[{now_iso()}] {label}")
    if data is not None:
        try:
            print(json.dumps(data, indent=2, ensure_ascii=False)[:2500])
        except Exception:
            print(data)


def stable_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def get_best_image_url(place: Optional[Dict[str, Any]]) -> str:
    if not place:
        return ""

    return pick_first_available(
        place.get("imageUrl"),
        place.get("wikiImage"),
        place.get("image"),
        place.get("photo"),
        place.get("photoUrl"),
        place.get("thumbnail"),
        fallback=""
    )


def has_image_url(place: Optional[Dict[str, Any]]) -> bool:
    image = get_best_image_url(place)
    return image.startswith("http://") or image.startswith("https://")

def pick_first_available(*values: Any, fallback: str = "") -> str:
    for value in values:
        text = safe_text(value)
        if text:
            return text
    return fallback


def get_message_text_from_history_item(item: Dict[str, Any]) -> str:
    return safe_text(
        item.get("content")
        or item.get("message")
        or item.get("reply")
        or item.get("text")
    )
#=============================================================
# DISTENCE FUNCTIONS
#=============================================================


def distance_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    try:
        r = 6371
        dlat = math.radians(lat2 - lat1)
        dlng = math.radians(lng2 - lng1)

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(dlng / 2) ** 2
        )

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c
    except Exception:
        return 999999


# ============================================================
# DOMAIN DATA
# ============================================================

KERALA_DISTRICTS = [
    "thiruvananthapuram", "trivandrum", "tvm", "kollam", "pathanamthitta",
    "alappuzha", "alleppey", "kottayam", "idukki", "ernakulam", "kochi",
    "cochin", "thrissur", "palakkad", "malappuram", "kozhikode", "calicut",
    "wayanad", "kannur", "kasaragod", "kasargod",
]

DISTRICT_ALIASES = {
    "trivandrum": "Thiruvananthapuram",
    "tvm": "Thiruvananthapuram",
    "alleppey": "Alappuzha",
    "calicut": "Kozhikode",
    "cochin": "Kochi",
    "ernakulam": "Kochi",
    "kasargod": "Kasaragod",
}

COMMON_PLACE_ALIASES = {
    "munar": "munnar",
    "munner": "munnar",
    "munnaar": "munnar",
    "alepy": "alleppey",
    "alappey": "alleppey",
    "alappuzha backwater": "alappuzha",
    "bekal": "bekal",
    "bekkal": "bekal",
    "beakal": "bekal",
    "varkala beach": "varkala",
    "kovalam beach": "kovalam",
    "athirapally": "athirappilly",
    "athirapilly": "athirappilly",
    "athirappally": "athirappilly",
    "waynad": "wayanad",
    "wagamon": "vagamon",
    "thekady": "thekkady",
    "gevi": "gavi",
}

STOPWORDS = set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "from",
    "for", "of", "in", "on", "at", "about", "tell", "me", "show", "give", "get",
    "with", "and", "or", "please", "need", "want", "image", "photo", "picture",
    "pic", "pics", "details", "detail", "place", "places", "kerala", "tour", "trip",
    "travel", "explain", "describe", "more", "one", "first", "second", "third",
])

MOOD_KEYWORDS = {
    "romantic": ["romantic", "couple", "honeymoon", "peaceful", "sunset", "private", "calm"],
    "family": ["family", "kids", "children", "safe", "picnic", "parents", "baby", "elder"],
    "adventure": ["adventure", "trek", "trekking", "hiking", "forest", "wildlife", "camping", "offroad"],
    "budget": ["budget", "cheap", "low cost", "affordable", "free", "less money"],
    "monsoon": ["rain", "monsoon", "rainy", "waterfall", "green"],
    "beach": ["beach", "coastal", "sea", "sunset", "shore", "lighthouse"],
    "hill": ["hill", "hill station", "munnar", "vagamon", "ponmudi", "tea", "misty", "cool"],
    "backwater": ["backwater", "lake", "boating", "houseboat", "canal", "kayaking"],
    "wildlife": ["wildlife", "forest", "sanctuary", "national park", "elephant", "tiger", "bird"],
    "heritage": ["heritage", "temple", "fort", "palace", "museum", "church", "mosque", "culture"],
    "food": ["food", "eat", "restaurant", "dish", "seafood", "sadya", "biriyani", "local food"],
}


INTENT_KEYWORDS = {
    "image": ["image", "photo", "picture", "pic", "pics", "photos", "pictures", "show image", "show photo", "image url", "photo url"],
    "travel_time": ["how much time", "how long", "time take", "travel time", "distance", "km", "from", "reach", "route", "drive", "bus", "train", "car", "bike", "scooter"],
    "recommendation": ["best", "top", "trending", "popular", "highest rated", "suggest", "recommend", "where to go", "places"],
    "trip_plan": ["plan", "trip", "itinerary", "2 day", "3 day", "one day", "1 day", "weekend", "schedule"],
    "best_time": ["when", "best time", "season", "month", "monsoon", "summer", "winter", "visit time"],
    "food": ["food", "eat", "restaurant", "local food", "dish", "breakfast", "lunch", "dinner"],
    "family_safety": ["safe", "family", "kids", "child", "children", "parents", "elder", "baby"],
    "compare": ["compare", "which is better", "better", " or ", " vs ", " versus "],
    "details": ["what about", "tell me about", "details", "information", "info", "about", "explain", "describe", "more about"],
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
# OSM / SERVICE DATA CONFIG
# ============================================================

OSM_COLLECTION = "osm_places"

OSM_TYPE_MAP = {
    "food": ["restaurant", "cafe", "fast_food", "food_court"],
    "osm_stay": ["hotel", "resort", "guest_house", "hostel"],
    "osm_health": ["hospital", "clinic", "doctors", "pharmacy", "dentist", "nursing_home"],
    "osm_transport": [
        "bus_stop", "bus_station", "station", "halt",
        "airport", "aerodrome", "terminal",
        "ferry_terminal", "taxi"
    ],
    "osm_emergency": ["police", "fire_station"],
    "osm_money": ["atm", "bank"],
    "osm_fuel": ["fuel", "charging_station"],
}

OSM_INTENT_KEYWORDS = {
    "osm_health": [
        "hospital", "hospitals", "clinic", "clinics", "doctor", "doctors",
        "pharmacy", "pharmacies", "medical", "medicine", "dentist"
    ],
    "osm_transport": [
        "airport", "airports", "railway", "railway station", "train station",
        "train", "bus stop", "bus station", "bus", "ferry", "taxi", "metro"
    ],
    "osm_money": [
        "atm", "bank", "banks"
    ],
    "osm_emergency": [
        "police", "police station", "fire station", "emergency"
    ],
    "osm_fuel": [
        "fuel", "petrol", "diesel", "ev charging", "charging station"
    ],
    "osm_stay": [
        "hotel", "hotels", "resort", "resorts", "stay", "guest house",
        "hostel", "room", "rooms", "accommodation"
    ],
}

KERALA_LOCATION_ALIASES = {
    "trivandrum": "thiruvananthapuram",
    "tvm": "thiruvananthapuram",
    "alleppey": "alappuzha",
    "calicut": "kozhikode",
    "cochin": "kochi",
    "ernakulam": "kochi",
    "kasargod": "kasaragod",
}

KERALA_LOCATIONS = [
    "thiruvananthapuram", "trivandrum", "tvm",
    "kollam", "pathanamthitta", "alappuzha", "alleppey",
    "kottayam", "idukki", "ernakulam", "kochi", "fort kochi",
    "thrissur", "palakkad", "malappuram", "kozhikode", "calicut",
    "wayanad", "kannur", "kasaragod", "kasargod",
    "munnar", "varkala", "kovalam", "bekal", "thekkady",
    "kumarakom", "nilambur", "kottakkal", "tirur", "ponnani",
    "perinthalmanna", "manjeri", "kalpetta", "sulthan bathery",
]

# ============================================================
# FIRESTORE NORMALIZATION / CACHE
# ============================================================

def normalize_place(doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    name = pick_first_available(
        data.get("name"),
        data.get("title"),
        fallback="Unknown Place"
    )

    description = pick_first_available(
        data.get("ai_description"),
        data.get("description"),
        data.get("desc"),
        data.get("about"),
        fallback="A beautiful destination in Kerala."
    )

    region = pick_first_available(
        data.get("region"),
        data.get("district"),
        data.get("category"),
        fallback="Kerala"
    )

    # Priority:
    # 1. old imageUrl
    # 2. wikiImage
    # 3. other old image fields
    image_url = pick_first_available(
        data.get("imageUrl"),
        data.get("wikiImage"),
        data.get("image"),
        data.get("photo"),
        data.get("photoUrl"),
        data.get("thumbnail"),
    )

    tags = split_tags(data.get("tags"))

    district = pick_first_available(
        data.get("district"),
        data.get("city"),
        data.get("location"),
        data.get("region")
    )

    category = pick_first_available(
        data.get("category"),
        data.get("type"),
        data.get("region")
    )

    aliases = split_tags(data.get("aliases"))
    generated_aliases = [name]

    name_norm = normalize_text(name)
    for wrong, correct in COMMON_PLACE_ALIASES.items():
        if correct in name_norm:
            generated_aliases.append(wrong)

    search_blob = " ".join([
        doc_id,
        name,
        region,
        district,
        category,
        safe_text(data.get("distance")),
        safe_text(data.get("bestTime")),
        description,
        " ".join(tags),
        " ".join(aliases),
        " ".join(generated_aliases),
    ])

    return {
        "id": doc_id,
        "name": name,
        "slug": slugify(name),
        "region": region,
        "district": district,
        "category": category,
        "distance": safe_text(data.get("distance"), ""),
        "imageUrl": image_url,
        "wikiImage": safe_text(data.get("wikiImage"), ""),
        "imageSource": safe_text(data.get("imageSource") or data.get("wikiImageSource"), ""),
        "rating": safe_float(data.get("rating"), 0.0),
        "userRatings": safe_int(data.get("userRatings") or data.get("reviews") or data.get("reviewCount"), 0),
        "ratingTotal": safe_float(data.get("ratingTotal"), 0.0),
        "bestTime": safe_text(data.get("bestTime") or data.get("best_time"), "Any time"),
        "description": description,
        "raw_description": safe_text(data.get("description"), ""),
        "ai_description": safe_text(data.get("ai_description"), ""),
        "tags": tags,
        "aliases": unique_list(aliases + generated_aliases),
        "source": safe_text(data.get("source"), ""),
        "latitude": safe_float(data.get("latitude") or data.get("lat"), 0.0),
        "longitude": safe_float(data.get("longitude") or data.get("lng") or data.get("lon"), 0.0),
        "search_blob": normalize_text(search_blob),
        "hasImage": bool(image_url),
    }

def build_places_index(places: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_id = {}
    by_slug = {}
    by_name = {}
    by_alias = {}
    with_images = []

    for place in places:
        place_id = safe_text(place.get("id"))
        name = safe_text(place.get("name"), "Unknown Place")

        # ✅ Fix old cache items missing slug
        slug = safe_text(place.get("slug"))
        if not slug:
            slug = slugify(name)
            place["slug"] = slug

        if place_id:
            by_id[place_id] = place

        if slug:
            by_slug[slug] = place

        name_key = normalize_text(name)
        if name_key:
            by_name[name_key] = place

        if has_image_url(place):
            with_images.append(place)

        aliases = place.get("aliases", [])
        if not isinstance(aliases, list):
            aliases = []

        for alias in aliases:
            alias_key = normalize_text(alias)
            if alias_key:
                by_alias[alias_key] = place

    return {
        "by_id": by_id,
        "by_slug": by_slug,
        "by_name": by_name,
        "by_alias": by_alias,
        "with_images": with_images,
        "builtAt": now_iso(),
    }

def save_places_to_local_cache(places: List[Dict[str, Any]]) -> None:
    try:
        payload = {
            "savedAt": now_iso(),
            "count": len(places),
            "places": places,
        }
        with open(LOCAL_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        debug_log("Saved places to local cache", {"file": LOCAL_CACHE_FILE, "count": len(places)})
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


def load_places_from_firestore(force: bool = False) -> List[Dict[str, Any]]:
    global PLACES_CACHE, PLACES_CACHE_TIME, PLACES_INDEX

    current_time = time.time()
    cache_valid = PLACES_CACHE and not force and (current_time - PLACES_CACHE_TIME) < CACHE_TTL_SECONDS

    if cache_valid:
        return PLACES_CACHE

    if not PLACES_CACHE and not force:
        local_places = load_places_from_local_cache()
        if local_places:
            PLACES_CACHE = local_places
            PLACES_CACHE_TIME = current_time
            PLACES_INDEX = build_places_index(PLACES_CACHE)
            return PLACES_CACHE

    places = []

    try:
        docs = db.collection(FIRESTORE_COLLECTION).stream()
        for doc in docs:
            data = doc.to_dict() or {}
            places.append(normalize_place(doc.id, data))

        PLACES_CACHE = places
        PLACES_CACHE_TIME = current_time
        PLACES_INDEX = build_places_index(PLACES_CACHE)
        save_places_to_local_cache(PLACES_CACHE)

        debug_log("Places cache refreshed from Firestore", {
            "collection": FIRESTORE_COLLECTION,
            "count": len(PLACES_CACHE),
            "withImages": len(PLACES_INDEX.get("with_images", [])),
        })
        return PLACES_CACHE

    except ResourceExhausted as e:
        debug_log("Firestore quota exceeded", str(e))
        if PLACES_CACHE:
            return PLACES_CACHE
        local_places = load_places_from_local_cache()
        if local_places:
            PLACES_CACHE = local_places
            PLACES_CACHE_TIME = current_time
            PLACES_INDEX = build_places_index(PLACES_CACHE)
            return PLACES_CACHE
        raise RuntimeError("Firestore quota exceeded and no local cache exists yet.")

    except GoogleAPIError as e:
        debug_log("Firestore Google API error", str(e))
        if PLACES_CACHE:
            return PLACES_CACHE
        local_places = load_places_from_local_cache()
        if local_places:
            PLACES_CACHE = local_places
            PLACES_CACHE_TIME = current_time
            PLACES_INDEX = build_places_index(PLACES_CACHE)
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
            PLACES_INDEX = build_places_index(PLACES_CACHE)
            return PLACES_CACHE
        raise


def get_place_by_id(place_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not place_id:
        return None
    places = load_places_from_firestore()
    if not PLACES_INDEX:
        globals()["PLACES_INDEX"] = build_places_index(places)
    return PLACES_INDEX.get("by_id", {}).get(place_id)


# ============================================================
# SEARCH / INTENT
# ============================================================

def apply_query_aliases(query: str) -> str:
    q = normalize_text(query)
    for wrong, correct in COMMON_PLACE_ALIASES.items():
        wrong_norm = normalize_text(wrong)
        correct_norm = normalize_text(correct)
        if wrong_norm in q and correct_norm not in q:
            q += " " + correct_norm
    return q


def tokenize(text: Any) -> List[str]:
    q = normalize_text(text)
    return [w for w in q.split() if len(w) >= 3 and w not in STOPWORDS]


def levenshtein_distance(a: str, b: str, max_distance: int = 3) -> int:
    a = normalize_text(a)
    b = normalize_text(b)

    if a == b:
        return 0

    if abs(len(a) - len(b)) > max_distance:
        return max_distance + 1

    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        row_min = i
        for j, cb in enumerate(b, 1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if ca == cb else 1)
            value = min(insert_cost, delete_cost, replace_cost)
            current.append(value)
            row_min = min(row_min, value)
        if row_min > max_distance:
            return max_distance + 1
        previous = current
    return previous[-1]


def fuzzy_ratio(a: str, b: str) -> float:
    a = normalize_text(a)
    b = normalize_text(b)
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    dist = levenshtein_distance(a, b, max_distance=max(3, int(max_len * 0.35)))
    return max(0.0, 1.0 - (dist / max_len))


def word_overlap_score(query: str, text: str) -> float:
    q_words = set(tokenize(query))
    t_words = set(tokenize(text))
    if not q_words or not t_words:
        return 0.0
    overlap = q_words.intersection(t_words)
    return len(overlap) / max(len(q_words), 1)


def contains_any(text: str, phrases: List[str]) -> bool:
    q = normalize_text(text)
    return any(normalize_text(p) in q for p in phrases)


def is_explicit_image_request(message: str) -> bool:
    q = " " + normalize_text(message) + " "
    image_words = [" image ", " photo ", " picture ", " pic ", " pics ", " photos ", " pictures "]
    return any(word in q for word in image_words)


def is_followup_message(message: str) -> bool:
    q = normalize_text(message)
    if not q:
        return False

    exact_followups = {
        "it", "this", "that", "there", "that place", "this place",
        "explain it", "explain this", "explain that", "more", "more details",
        "tell more", "about it", "details", "full details", "best time",
        "how to go", "route", "distance", "pic one", "photo one", "image one",
        "first one", "second one", "third one", "1st one", "2nd one", "3rd one",
    }

    if q in exact_followups:
        return True

    starts = [
        "explain", "describe", "tell more", "more about", "about this", "about that",
        "how to", "best time", "pic", "photo", "image", "show pic", "show photo",
    ]
    return any(q.startswith(s) for s in starts)


def detect_selected_index(message: str) -> Optional[int]:
    q = normalize_text(message)

    word_map = {
        "one": 0, "first": 0, "1st": 0, "1": 0,
        "two": 1, "second": 1, "2nd": 1, "2": 1,
        "three": 2, "third": 2, "3rd": 2, "3": 2,
        "four": 3, "fourth": 3, "4th": 3, "4": 3,
        "five": 4, "fifth": 4, "5th": 4, "5": 4,
        "six": 5, "sixth": 5, "6th": 5, "6": 5,
        "seven": 6, "seventh": 6, "7th": 6, "7": 6,
    }

    for word, index in word_map.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return index

    return None


def detect_intent(message: str) -> str:
    q = " " + normalize_text(message) + " "

    if is_explicit_image_request(message):
        return "image"

    if contains_any(q, INTENT_KEYWORDS["travel_time"]) and (
        " from " in q or " to " in q or "distance" in q or "route" in q
    ):
        return "travel_time"

    if contains_any(q, INTENT_KEYWORDS["compare"]):
        if re.search(r"\s+or\s+|\s+vs\s+|\s+versus\s+|which is better", q):
            return "compare"

    # OSM/service intents should come before recommendation/details
    for osm_intent, keywords in OSM_INTENT_KEYWORDS.items():
        if contains_any(q, keywords):
            return osm_intent

    if contains_any(q, INTENT_KEYWORDS["trip_plan"]):
        return "trip_plan"

    # Existing food intent:
    # "best Kerala food" -> normal food answer
    # "restaurant in Munnar" -> OSM because food place query
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

def is_food_place_query(message: str) -> bool:
    q = normalize_text(message)

    place_words = [
        "restaurant", "restaurants", "cafe", "cafes", "hotel",
        "near", "nearby", "in ", "at ", "around",
        "where to eat", "food spot", "food spots", "eat"
    ]

    return any(w in q for w in place_words)
def extract_day_count(message: str) -> int:
    q = normalize_text(message)
    for pattern in [r"(\d+)\s*day", r"(\d+)\s*days"]:
        m = re.search(pattern, q)
        if m:
            return max(1, min(10, int(m.group(1))))
    if "weekend" in q:
        return 2
    if "one day" in q or "1 day" in q:
        return 1
    return 2


def extract_requested_count(message: str, default: int = 7) -> int:
    q = normalize_text(message)
    m = re.search(r"\b(\d{1,2})\b", q)
    if m:
        return max(1, min(20, int(m.group(1))))
    return default


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


def place_search_score(query: str, place: Dict[str, Any]) -> float:
    q = apply_query_aliases(query)
    name = normalize_text(place.get("name", ""))
    slug = normalize_text(place.get("slug", ""))
    region = normalize_text(place.get("region", ""))
    district = normalize_text(place.get("district", ""))
    category = normalize_text(place.get("category", ""))
    distance = normalize_text(place.get("distance", ""))
    description = normalize_text(place.get("description", ""))
    tags = " ".join(normalize_text(t) for t in place.get("tags", []))
    aliases = " ".join(normalize_text(t) for t in place.get("aliases", []))
    search_blob = place.get("search_blob", "")

    score = 0.0

    if not q:
        return score

    if name and name in q:
        score += 180
    if q and q in name:
        score += 130
    if slug and slug in q:
        score += 120

    for alias in place.get("aliases", []):
        alias_norm = normalize_text(alias)
        if alias_norm and alias_norm in q:
            score += 110

    q_tokens = tokenize(q)
    name_tokens = set(tokenize(name))
    region_tokens = set(tokenize(region))
    district_tokens = set(tokenize(district))
    tag_tokens = set(tokenize(tags))
    category_tokens = set(tokenize(category))

    for token in q_tokens:
        if token in name_tokens:
            score += 38
        if token in region_tokens:
            score += 16
        if token in district_tokens:
            score += 18
        if token in tag_tokens:
            score += 14
        if token in category_tokens:
            score += 12
        if token in distance:
            score += 5
        if token in description:
            score += 4
        if token in aliases:
            score += 22
        if token in search_blob:
            score += 3

    if len(q_tokens) <= 5:
        query_without_noise = " ".join(q_tokens)
        ratio = fuzzy_ratio(query_without_noise, name)
        if ratio >= 0.70:
            score += ratio * 80

    moods = extract_moods(query)
    for mood in moods:
        mood_words = MOOD_KEYWORDS.get(mood, [])
        text_blob = f"{name} {region} {district} {category} {tags} {description}"
        if any(normalize_text(w) in text_blob for w in mood_words):
            score += 30

    districts = extract_districts(query)
    for district_name in districts:
        d = normalize_text(district_name)
        if d in district or d in region or d in distance or d in description or d in tags:
            score += 25

    score += word_overlap_score(q, search_blob) * 45

    rating = safe_float(place.get("rating"), 0)
    reviews = safe_int(place.get("userRatings"), 0)
    score += rating * 1.8
    score += min(reviews / 150, 8)

    if has_image_url(place):
        score += 2

    return score


def search_places(query: str, places: List[Dict[str, Any]], limit: int = 8, min_score: float = 5.0, require_image: bool = False) -> List[Dict[str, Any]]:
    scored = []
    for place in places:
        if require_image and not has_image_url(place):
            continue
        score = place_search_score(query, place)
        if score >= min_score:
            p = dict(place)
            p["_score"] = round(score, 2)
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:limit]]
def extract_location_keyword(message: str) -> str:
    q = normalize_text(message)

    for loc in KERALA_LOCATIONS:
        if loc in q:
            return KERALA_LOCATION_ALIASES.get(loc, loc)

    return ""


def get_best_osm_image(data: Dict[str, Any]) -> str:
    return pick_first_available(
        data.get("imageUrl"),
        data.get("wikiImage"),
        fallback=""
    )


def build_osm_description(data: Dict[str, Any]) -> str:
    name = safe_text(data.get("name"), "This place")
    place_type = safe_text(data.get("type"), "place").replace("_", " ")
    city = safe_text(data.get("city") or data.get("district") or "Kerala")
    address = safe_text(data.get("address"))
    phone = safe_text(data.get("phone"))
    opening = safe_text(data.get("openingHours"))

    text = f"{name} is a {place_type} located in {city}."

    if address:
        text += f" Address: {address}."
    if phone:
        text += f" Phone: {phone}."
    if opening:
        text += f" Opening hours: {opening}."

    text += " Data source: OpenStreetMap."
    return text


def normalize_osm_card(doc_id: str, data: Dict[str, Any], score: float = 0) -> Dict[str, Any]:
    loc = data.get("location") or {}

    lat = safe_float(
        loc.get("lat") if isinstance(loc, dict) else data.get("latitude"),
        0.0
    )

    lng = safe_float(
        loc.get("lng") if isinstance(loc, dict) else data.get("longitude"),
        0.0
    )

    return {
        "id": doc_id,
        "osmId": data.get("osmId"),
        "name": data.get("name"),
        "slug": slugify(safe_text(data.get("name"))),
        "type": data.get("type"),
        "category": data.get("category"),
        "region": data.get("district") or data.get("city") or data.get("state") or "Kerala",
        "district": data.get("district"),
        "city": data.get("city"),
        "address": data.get("address"),
        "phone": data.get("phone"),
        "website": data.get("website"),
        "openingHours": data.get("openingHours"),
        "location": loc,

        "lat": lat,
        "lng": lng,
        "latitude": lat,
        "longitude": lng,

        "imageUrl": get_best_osm_image(data),
        "wikiImage": data.get("wikiImage"),
        "source": "OpenStreetMap",
        "isOsm": True,
        "rating": 0,
        "userRatings": 0,
        "bestTime": data.get("openingHours") or "Available on map",
        "distance": data.get("address") or data.get("city") or data.get("district") or "Kerala",
        "description": build_osm_description(data),
        "tags": [],
        "_score": round(score, 2),
    }
     

def build_searchable_osm_text(data: Dict[str, Any]) -> str:
    keywords = data.get("searchKeywords", [])
    if not isinstance(keywords, list):
        keywords = []

    tags = data.get("tags", {})
    tag_text = ""

    if isinstance(tags, dict):
        tag_text = " ".join([str(v) for v in tags.values() if v])

    return " ".join([
        safe_text(data.get("name")),
        safe_text(data.get("type")),
        safe_text(data.get("category")),
        safe_text(data.get("district")),
        safe_text(data.get("city")),
        safe_text(data.get("state")),
        safe_text(data.get("address")),
        safe_text(data.get("cuisine")),
        safe_text(data.get("operator")),
        safe_text(data.get("brand")),
        " ".join(map(str, keywords)),
        tag_text,
    ]).lower()


def score_osm_place(data: Dict[str, Any], message: str, intent: str, location: str) -> float:
    q = normalize_text(message)
    searchable = build_searchable_osm_text(data)

    place_type = safe_text(data.get("type")).lower()
    category = safe_text(data.get("category")).lower()
    allowed_types = OSM_TYPE_MAP.get(intent, [])

    score = 0

    if allowed_types and place_type in allowed_types:
        score += 70

    if location and location in searchable:
        score += 60

    if category and category in searchable:
        score += 10

    for word in q.split():
        if len(word) >= 4 and word in searchable:
            score += 8

    if data.get("name"):
        score += 10
    if data.get("address"):
        score += 4
    if data.get("phone"):
        score += 4
    if data.get("website"):
        score += 4
    if data.get("openingHours"):
        score += 4

    return score


def search_osm_places(
    message: str,
    intent: str,
    limit: int = 8,
    user_lat: float = 0.0,
    user_lng: float = 0.0,
) -> List[Dict[str, Any]]:
    location = extract_location_keyword(message)
    allowed_types = OSM_TYPE_MAP.get(intent, [])

    candidates = []

    try:
        # Best case: search by location keyword
        if location:
            docs = (
                db.collection(OSM_COLLECTION)
                .where("searchKeywords", "array_contains", location)
                .limit(1000)
                .stream()
            )
        else:
            # Safe fallback. Do not fetch full collection.
            docs = (
                db.collection(OSM_COLLECTION)
                .limit(1500)
                .stream()
            )

        for doc in docs:
            data = doc.to_dict() or {}
            place_type = safe_text(data.get("type")).lower()
            loc = data.get("location") or {}
            place_lat = safe_float(loc.get("lat") if isinstance(loc, dict) else data.get("latitude"), 0.0)
            place_lng = safe_float(loc.get("lng") if isinstance(loc, dict) else data.get("longitude"), 0.0)

            nearby_distance = None
            if user_lat and user_lng and place_lat and place_lng:
                nearby_distance = distance_km(user_lat, user_lng, place_lat, place_lng)

            if allowed_types and place_type not in allowed_types:
                continue

            score = score_osm_place(data, message, intent, location)
            if nearby_distance is not None:
                if nearby_distance <= 2:
                    score += 120
                elif nearby_distance <= 5:
                    score += 90
                elif nearby_distance <= 10:
                    score += 60
                elif nearby_distance <= 25:
                    score += 30
                else:
                    score -= min(nearby_distance, 200) * 0.5

            if score > 0:
                card = normalize_osm_card(doc.id, data, score)
                if nearby_distance is not None:
                    card["distanceKm"] = round(nearby_distance, 2)
                candidates.append(card)

    except Exception as e:
        debug_log("OSM search failed", {"error": str(e), "message": message, "intent": intent})
        return []

    if user_lat and user_lng:
        candidates.sort(key=lambda p: (p.get("distanceKm", 999999), -p.get("_score", 0)))
    else:
        candidates.sort(key=lambda p: p.get("_score", 0), reverse=True)
    return candidates[:limit]

def trending_score(place: Dict[str, Any]) -> float:
    rating = safe_float(place.get("rating"), 0)
    reviews = safe_int(place.get("userRatings"), 0)
    image_bonus = 12 if has_image_url(place) else 0
    return (rating * 20) + min(reviews * 1.5, 500) + image_bonus


def get_trending_places(places: List[Dict[str, Any]], limit: int = 8, require_image: bool = False) -> List[Dict[str, Any]]:
    filtered = [p for p in places if not require_image or has_image_url(p)]
    return sorted(filtered, key=trending_score, reverse=True)[:limit]


def find_place_by_name(query: str, places: List[Dict[str, Any]], min_score: float = 50) -> Optional[Dict[str, Any]]:
    q = apply_query_aliases(query)

    if not PLACES_INDEX:
        globals()["PLACES_INDEX"] = build_places_index(places)

    index = PLACES_INDEX
    q_norm = normalize_text(q)

    if q_norm in index.get("by_name", {}):
        return index["by_name"][q_norm]
    if q_norm in index.get("by_slug", {}):
        return index["by_slug"][q_norm]
    if q_norm in index.get("by_alias", {}):
        return index["by_alias"][q_norm]

    for place in places:
        name = normalize_text(place.get("name", ""))
        if name and name in q_norm:
            return place
        for alias in place.get("aliases", []):
            alias_norm = normalize_text(alias)
            if alias_norm and alias_norm in q_norm:
                return place

    best = None
    best_score = 0.0
    for place in places:
        score = place_search_score(q, place)
        if score > best_score:
            best = place
            best_score = score

    if best and best_score >= min_score:
        return best
    return None


def find_multiple_places_in_message(message: str, places: List[Dict[str, Any]], limit: int = 4) -> List[Dict[str, Any]]:
    q = apply_query_aliases(message)
    found = []
    found_ids = set()

    for place in places:
        name = normalize_text(place.get("name", ""))
        aliases = [normalize_text(a) for a in place.get("aliases", [])]
        if (name and name in q) or any(a and a in q for a in aliases):
            found.append(place)
            found_ids.add(place["id"])

    searched = search_places(q, places, limit=limit * 2, min_score=35)
    for place in searched:
        if place["id"] not in found_ids:
            found.append(place)
            found_ids.add(place["id"])
        if len(found) >= limit:
            break

    return found[:limit]


def extract_context_place_from_history(history: List[Dict[str, Any]], places: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(history, list):
        return None

    for item in reversed(history[-10:]):
        place_id = safe_text(item.get("currentPlaceId") or item.get("placeId"))
        if place_id:
            place = get_place_by_id(place_id)
            if place:
                return place

        content = get_message_text_from_history_item(item)
        if not content:
            continue

        found = find_place_by_name(content, places, min_score=45)
        if found:
            return found

    return None


def select_place_from_previous_list(
    selected_index: Optional[int],
    last_matched_place_ids: List[str],
    current_place: Optional[Dict[str, Any]],
    places: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if selected_index is not None and last_matched_place_ids:
        if 0 <= selected_index < len(last_matched_place_ids):
            selected = get_place_by_id(last_matched_place_ids[selected_index])
            if selected:
                return selected

    if current_place:
        return current_place

    return None


# ============================================================
# IMAGE / UI SUPPORT
# ============================================================

def is_image_request(message: str) -> bool:
    return detect_intent(message) == "image"


def build_place_card(
    place: Dict[str, Any],
    include_description: bool = False,
    include_image: bool = True
) -> Dict[str, Any]:

    best_image = get_best_image_url(place)

    location = place.get("location") or {}
    location_lat = location.get("lat") if isinstance(location, dict) else None
    location_lng = location.get("lng") if isinstance(location, dict) else None

    lat = safe_float(
        place.get("lat")
        or place.get("latitude")
        or location_lat,
        0.0
    )

    lng = safe_float(
        place.get("lng")
        or place.get("lon")
        or place.get("longitude")
        or location_lng,
        0.0
    )

    if place.get("isOsm"):
        card = {
            "id": place.get("id"),
            "osmId": place.get("osmId"),
            "name": place.get("name"),
            "slug": place.get("slug"),
            "region": place.get("region"),
            "district": place.get("district"),
            "city": place.get("city"),
            "category": place.get("category"),
            "type": place.get("type"),
            "address": place.get("address"),
            "phone": place.get("phone"),
            "website": place.get("website"),
            "openingHours": place.get("openingHours"),
            "distanceKm": place.get("distanceKm"),

            # ✅ Important for frontend map
            "lat": lat,
            "lng": lng,
            "latitude": lat,
            "longitude": lng,

            "rating": 0,
            "userRatings": 0,
            "bestTime": place.get("bestTime"),
            "distance": place.get("distance"),
            "imageUrl": best_image if include_image else None,
            "wikiImage": place.get("wikiImage"),
            "hasImage": has_image_url(place) if include_image else False,
            "tags": [],
            "score": place.get("_score"),
            "source": "OpenStreetMap",
            "isOsm": True,
        }

        if include_description:
            card["description"] = clean_for_display(place.get("description", ""), 500)

        return card

    card = {
        "id": place.get("id"),
        "name": place.get("name"),
        "slug": place.get("slug"),
        "region": place.get("region"),
        "district": place.get("district"),
        "category": place.get("category"),

        # ✅ Important for frontend map
        "lat": lat,
        "lng": lng,
        "latitude": lat,
        "longitude": lng,

        "rating": safe_float(place.get("rating"), 0),
        "userRatings": safe_int(place.get("userRatings"), 0),
        "bestTime": place.get("bestTime"),
        "distance": place.get("distance"),
        "imageUrl": best_image if include_image else None,
        "wikiImage": place.get("wikiImage"),
        "hasImage": has_image_url(place) if include_image else False,
        "tags": place.get("tags", []),
        "score": place.get("_score"),
    }

    if include_description:
        card["description"] = clean_for_display(place.get("description", ""), 500)

    return card


def build_ui_directives(
    intent: str,
    message: str,
    primary_place: Optional[Dict[str, Any]],
    matched_count: int,
    is_followup: bool,
    requested_count: int,
) -> Dict[str, Any]:
    explicit_image = is_explicit_image_request(message)

    if intent == "image" and explicit_image:
        return {
            "showImages": True,
            "showBigImage": True,
            "showCards": True,
            "cardMode": "image_result",
            "cardLimit": 1,
            "reason": "user_asked_for_image",
        }

    if intent in ["recommendation", "trip_plan"]:
        limit = min(DEFAULT_CARD_LIMIT, requested_count, max(matched_count, 1))
        return {
            "showImages": True,
            "showBigImage": False,
            "showCards": True,
            "cardMode": "list_with_images",
            "cardLimit": limit,
            "reason": "list_answer_cards_are_useful",
        }

    if intent == "compare":
        return {
            "showImages": True,
            "showBigImage": False,
            "showCards": True,
            "cardMode": "compare_cards",
            "cardLimit": 2,
            "reason": "comparison_cards",
        }

    if intent in OSM_TYPE_MAP:
        return {
            "showImages": False,
            "showBigImage": False,
            "showCards": True,
            "cardMode": "osm_service_cards",
            "cardLimit": min(DEFAULT_CARD_LIMIT, requested_count, max(matched_count, 1)),
            "reason": "osm_service_results_one_best_card",
        }

    if primary_place:
        return {
            "showImages": True,
            "showBigImage": False,
            "showCards": True,
            "cardMode": "compact_place_with_image",
            "cardLimit": EXPLAIN_CARD_LIMIT,
            "reason": "followup_place_card" if is_followup else "specific_place_card",
        }

    return {
        "showImages": False,
        "showBigImage": False,
        "showCards": False,
        "cardMode": "text_only",
        "cardLimit": 0,
        "reason": "text_answer_only",
    }
def build_media_response(place: Optional[Dict[str, Any]], alternatives: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    alternatives = alternatives or []

    if place and has_image_url(place):
        return {
            "type": "image",
            "status": "found",
            "title": place.get("name"),
            "imageUrl": get_best_image_url(place),
            "placeId": place.get("id"),
            "place": build_place_card(place, include_description=True, include_image=True),
            "alternatives": [
                build_place_card(p, include_image=True)
                for p in alternatives
                if p.get("id") != place.get("id")
            ][:4],
        }

    return {
        "type": "image",
        "status": "not_found",
        "title": None,
        "imageUrl": None,
        "placeId": None,
        "place": None,
        "alternatives": [build_place_card(p, include_image=True) for p in alternatives[:4]],
    }


def reply_for_image_request(place: Optional[Dict[str, Any]], alternatives: List[Dict[str, Any]]) -> str:
    image_url = get_best_image_url(place)

    if place and image_url:
        return (
            f"Here is the image for {place['name']} 🌴\n\n"
            f"Image URL: {image_url}\n\n"
            f"Best time: {place.get('bestTime', 'Any time')}\n"
            f"Location/category: {place.get('region', 'Kerala')}"
        )

    if place and not image_url:
        return f"I found {place['name']}, but this place does not have an image saved in Firebase yet."

    if alternatives:
        names = ", ".join([p.get("name", "Unknown") for p in alternatives[:3]])
        return f"I could not find an exact image for that place, but I found possible matches: {names}."

    return "I could not find that place image in your Firebase data yet."

# ============================================================
# TRAIN / RAILWAY SERVICE
# ============================================================

NTES_URL = "https://enquiry.indianrail.gov.in/ntes/"

DISTRICT_RAIL_ALIASES = {
    "trivandrum": "thiruvananthapuram",
    "thiruvananthapuram": "thiruvananthapuram",
    "tvm": "thiruvananthapuram",

    "kollam": "kollam",
    "pathanamthitta": "pathanamthitta",

    "alappuzha": "alappuzha",
    "alleppey": "alappuzha",

    "kottayam": "kottayam",
    "idukki": "idukki",

    "ernakulam": "ernakulam",
    "kochi": "ernakulam",
    "cochin": "ernakulam",

    "thrissur": "thrissur",
    "trichur": "thrissur",

    "palakkad": "palakkad",
    "palghat": "palakkad",

    "malappuram": "malappuram",
    "malapuram": "malappuram",
    "manjeri": "malappuram",
    "perinthalmanna": "malappuram",
    "kottakkal": "malappuram",
    "tirur": "malappuram",
    "nilambur": "malappuram",

    "kozhikode": "kozhikode",
    "calicut": "kozhikode",

    "wayanad": "wayanad",
    "kannur": "kannur",

    "kasaragod": "kasaragod",
    "kasargod": "kasaragod",
}

STATION_ALIASES = {
    # Ernakulam / Kochi
    "kochi": "ERS",
    "cochin": "ERS",
    "ernakulam": "ERS",
    "ernakulam junction": "ERS",
    "ernakulam south": "ERS",
    "ernakulam town": "ERN",
    "ernakulam north": "ERN",
    "aluva": "AWY",
    "alwaye": "AWY",
    "angamaly": "AFK",

    # South Kerala
    "trivandrum": "TVC",
    "thiruvananthapuram": "TVC",
    "thiruvananthapuram central": "TVC",
    "tvm": "TVC",
    "kollam": "QLN",
    "quilon": "QLN",
    "varkala": "VAK",

    # Alappuzha / Kottayam
    "alappuzha": "ALLP",
    "alleppey": "ALLP",
    "kayamkulam": "KYJ",
    "kottayam": "KTYM",
    "chengannur": "CNGR",
    "tiruvalla": "TRVL",

    # Central Kerala
    "thrissur": "TCR",
    "trichur": "TCR",
    "guruvayur": "GUV",
    "chalakudi": "CKI",
    "shoranur": "SRR",
    "palakkad": "PGT",
    "palakkad junction": "PGT",
    "palakkad town": "PGTN",

    # Malappuram
    "malappuram": "malappuram",
    "malapuram": "malappuram",
    "manjeri": "malappuram",
    "perinthalmanna": "malappuram",
    "kottakkal": "malappuram",
    "angadipuram": "AAM",
    "tirur": "TIR",
    "kuttippuram": "KTU",
    "parappanangadi": "PGI",
    "nilambur": "NIL",
    "nilambur road": "NIL",

    # North Kerala
    "kozhikode": "CLT",
    "calicut": "CLT",
    "ferok": "FK",
    "vadakara": "BDJ",
    "kannur": "CAN",
    "thalassery": "TLY",
    "payyanur": "PAY",
    "kasaragod": "KGQ",
    "kasargod": "KGQ",
    "bekal": "bekal",
    "bekal fort": "BFR",
    "kanhangad": "KZE",
    "manjeshwar": "MJS",

    # Tourist places without direct major railway
    "munnar": "munnar",
    "wayanad": "wayanad",
    "vagamon": "vagamon",
    "thekkady": "thekkady",
    "athirappilly": "athirappilly",
    "athirapally": "athirappilly",
    "athirapilly": "athirappilly",
}


def is_train_question(message: str) -> bool:
    msg = (message or "").lower()

    strong_train_words = [
        "train",
        "railway",
        "railways",
        "rail",
        "station",
        "by train",
        "in train",
        "train route",
        "train time",
        "train timing",
        "nearest railway station",
        "nearest station",
        "kerala railway",
        "kerala railways",
    ]

    return any(word in msg for word in strong_train_words)


def extract_train_districts(message: str):
    msg = (message or "").lower()
    found = []

    sorted_aliases = sorted(
        DISTRICT_RAIL_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, district_key in sorted_aliases:
        if alias in msg and district_key not in found:
            found.append(district_key)

    return found


def extract_station_or_place_codes(message: str):
    msg = (message or "").lower()
    found = []

    sorted_aliases = sorted(
        STATION_ALIASES.items(),
        key=lambda x: len(x[0]),
        reverse=True
    )

    for alias, code in sorted_aliases:
        if alias in msg and code not in found:
            found.append(code)

    return found


def get_district_railway_access(district_key: str):
    doc = db.collection("districtRailwayAccess").document(district_key).get()
    if doc.exists:
        return doc.to_dict()
    return None


def get_station_by_code(code: str):
    doc = db.collection("railwayStations").document(code).get()
    if doc.exists:
        return doc.to_dict()
    return None


def get_route(from_code: str, to_code: str):
    route_id = f"{from_code}_{to_code}"
    doc = db.collection("trainRoutes").document(route_id).get()
    if doc.exists:
        return doc.to_dict()
    return None


def get_nearest_station_for_place(place_key: str):
    doc = db.collection("nearestRailwayStations").document(place_key).get()
    if doc.exists:
        return doc.to_dict()
    return None


def make_train_response(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Your normal backend uses 'reply'.
    Train helper used 'answer'.
    This function returns both, so frontend will work.
    """
    answer = payload.get("answer", "")
    payload["reply"] = answer
    payload["answer"] = answer
    payload["aiUsed"] = False
    payload["ui"] = {
        "showCards": False,
        "showBigImage": False,
        "showImages": False,
        "cardMode": "train_text",
        "cardLimit": 0,
    }
    payload["matchedPlaces"] = []
    payload["media"] = None
    payload["timestamp"] = now_iso()
    payload["requestId"] = stable_hash(answer + str(time.time()))
    return payload


def handle_train_question(message: str):
    msg = (message or "").lower().strip()

    source = {
        "title": "National Train Enquiry System",
        "source": "Indian Railways",
        "link": NTES_URL
    }

    tourist_place_keys = [
        "munnar",
        "wayanad",
        "vagamon",
        "thekkady",
        "bekal",
        "athirappilly",
        "malappuram",
    ]

    # -------------------------------------------------
    # Case 0: General Kerala railway overview
    # -------------------------------------------------
    if (
        "kerala railway" in msg
        or "kerala railways" in msg
        or "about railway" in msg
        or "about kerala railway" in msg
        or msg.strip() in ["railway", "train", "kerala train", "kerala railway"]
    ):
        highlights_doc = db.collection("keralaRailwayStationList").document("highlights").get()

        if highlights_doc.exists:
            highlights = highlights_doc.to_dict()

            important = highlights.get("importantStations", [])
            important_text = ", ".join(important[:8]) if important else (
                "Thiruvananthapuram Central, Kollam Junction, Ernakulam Junction, "
                "Thrissur, Shoranur Junction, Palakkad Junction, Kozhikode, Kannur, Kasaragod"
            )

            no_rail = highlights.get("districtsWithoutRailway", [])
            no_rail_text = ", ".join(no_rail) if no_rail else "Wayanad and Idukki"

            answer = (
                "Kerala has a strong railway network running from Thiruvananthapuram in the south "
                "to Kasaragod in the north. It connects major travel hubs like Kollam, Alappuzha, "
                "Kottayam, Ernakulam, Thrissur, Palakkad, Kozhikode, Kannur, and Kasaragod.\n\n"
                f"Important stations include: {important_text}.\n\n"
                f"Districts without direct railway stations: {no_rail_text}.\n\n"
                "For tourism, trains are useful for reaching Kochi, Alappuzha, Kozhikode, Kannur, "
                "Bekal, Varkala, Kollam, Thrissur, and Palakkad. Hill destinations like Munnar, "
                "Wayanad, Vagamon, and Thekkady usually need train + road travel.\n\n"
                "For live train timing, please verify on NTES."
            )

            return make_train_response({
                "success": True,
                "intent": "kerala_railway_overview",
                "answer": answer,
                "data": highlights,
                "sources": [source],
            })

        return make_train_response({
            "success": True,
            "intent": "kerala_railway_overview",
            "answer": (
                "Kerala has a major railway network across South, Central, and North Kerala. "
                "Main stations include Thiruvananthapuram Central, Kollam Junction, Ernakulam Junction, "
                "Thrissur, Shoranur Junction, Palakkad Junction, Kozhikode, Kannur, and Kasaragod. "
                "For live train timing, please check NTES."
            ),
            "sources": [source],
        })

    codes = extract_station_or_place_codes(message)
    districts = extract_train_districts(message)

    # -------------------------------------------------
    # Case 1: List all Kerala railway stations
    # -------------------------------------------------
    if (
        "list" in msg
        or "all railway" in msg
        or "railway stations in kerala" in msg
        or "all stations" in msg
    ):
        sections = db.collection("keralaRailwayStationList").stream()

        railway_sections = []

        for doc in sections:
            d = doc.to_dict()
            if doc.id != "highlights":
                railway_sections.append({
                    "id": doc.id,
                    "title": d.get("title"),
                    "routeType": d.get("routeType"),
                    "description": d.get("description"),
                    "stations": d.get("stations", []),
                    "stationCount": len(d.get("stations", [])),
                })

        answer = (
            "Kerala railway stations can be viewed by regions and routes: "
            "Thiruvananthapuram, Kollam, Kayamkulam-Ernakulam via Kottayam, "
            "Kayamkulam-Ernakulam via Alappuzha, Kochi/Central Kerala, Malabar, Kasaragod, "
            "Shoranur-Palakkad, Shoranur-Nilambur, Kollam-Sengottai, Palakkad-Pollachi, "
            "and Thrissur-Guruvayur.\n\n"
            "For live train timing, always verify on NTES."
        )

        return make_train_response({
            "success": True,
            "intent": "kerala_railway_station_list",
            "answer": answer,
            "sections": railway_sections,
            "sources": [source],
        })

    # -------------------------------------------------
    # Case 2: District to district railway guidance
    # Example: "I want to go Malappuram from Ernakulam in train"
    # -------------------------------------------------
    if len(districts) >= 2:
        from_district_key = districts[0]
        to_district_key = districts[1]

        # Handle "go X from Y"
        if " from " in msg:
            before_from, after_from = msg.split(" from ", 1)

            detected_before = extract_train_districts(before_from)
            detected_after = extract_train_districts(after_from)

            if detected_before and detected_after:
                to_district_key = detected_before[0]
                from_district_key = detected_after[0]

        from_district = get_district_railway_access(from_district_key)
        to_district = get_district_railway_access(to_district_key)

        if from_district and to_district:
            from_stations = from_district.get("mainStations", [])
            to_stations = to_district.get("mainStations", [])

            from_station = from_stations[0] if from_stations else None

            to_station_lines = []
            for s in to_stations[:5]:
                to_station_lines.append(
                    f"- {s.get('name')} ({s.get('code')}): {s.get('bestFor')}"
                )

            station_text = "\n".join(to_station_lines)

            if from_station:
                start_text = (
                    f"From {from_district.get('district')}, you can start from "
                    f"{from_station.get('name')} ({from_station.get('code')})."
                )
            else:
                start_text = f"From {from_district.get('district')}, choose the nearest available railway station."

            if to_district.get("hasRailway") is False:
                rail_note = (
                    f"{to_district.get('district')} does not have a railway station. "
                    f"Use one of the nearest useful stations below and continue by road."
                )
            else:
                rail_note = f"For {to_district.get('district')}, useful railway options are:"

            answer = (
                f"{start_text}\n\n"
                f"{rail_note}\n"
                f"{station_text}\n\n"
                f"{to_district.get('travelNote')} "
                f"Train timings may change, so please check latest timing on NTES before travel."
            )

            return make_train_response({
                "success": True,
                "intent": "district_train_access",
                "answer": answer,
                "fromDistrict": from_district,
                "toDistrict": to_district,
                "sources": [source],
            })

    # -------------------------------------------------
    # Case 3: One district railway guidance
    # -------------------------------------------------
    if len(districts) == 1:
        district_key = districts[0]
        district_data = get_district_railway_access(district_key)

        if district_data:
            stations = district_data.get("mainStations", [])

            station_lines = []
            for s in stations[:5]:
                station_lines.append(
                    f"- {s.get('name')} ({s.get('code')}): {s.get('bestFor')}"
                )

            station_text = "\n".join(station_lines)

            if district_data.get("hasRailway") is False:
                intro = f"{district_data.get('district')} does not have a railway station."
            else:
                intro = f"{district_data.get('district')} has railway access."

            answer = (
                f"{intro}\n\n"
                f"Useful railway options:\n"
                f"{station_text}\n\n"
                f"{district_data.get('travelNote')} "
                f"For live timing, please check NTES."
            )

            return make_train_response({
                "success": True,
                "intent": "district_railway_access",
                "answer": answer,
                "data": district_data,
                "sources": [source],
            })

    # -------------------------------------------------
    # Case 4: Tourist place nearest station
    # -------------------------------------------------
    for code in codes:
        if code in tourist_place_keys:
            place_data = get_nearest_station_for_place(code)

            if place_data:
                answer = (
                    f"{place_data.get('placeName')} does not have a direct major railway station. "
                    f"The nearest useful railway station is {place_data.get('nearestMajorStationName')} "
                    f"({place_data.get('nearestMajorStationCode')}). "
                    f"{place_data.get('travelNote')} "
                    f"Approx road travel: {place_data.get('approxRoadTimeFromStation')}. "
                    f"For latest train timing, please verify on NTES."
                )

                return make_train_response({
                    "success": True,
                    "intent": "train_nearest_station",
                    "answer": answer,
                    "data": place_data,
                    "sources": [source],
                })

    station_codes = [
        c for c in codes
        if c not in tourist_place_keys
    ]

    # -------------------------------------------------
    # Case 5: Station to station route
    # -------------------------------------------------
    if len(station_codes) >= 2:
        from_code = station_codes[0]
        to_code = station_codes[1]

        if " from " in msg:
            before_from, after_from = msg.split(" from ", 1)

            before_codes = extract_station_or_place_codes(before_from)
            after_codes = extract_station_or_place_codes(after_from)

            before_station_codes = [c for c in before_codes if c not in tourist_place_keys]
            after_station_codes = [c for c in after_codes if c not in tourist_place_keys]

            if before_station_codes and after_station_codes:
                to_code = before_station_codes[0]
                from_code = after_station_codes[0]

        route = get_route(from_code, to_code)

        if route:
            answer = (
                f"You can travel from {route.get('fromName')} ({route.get('fromCode')}) "
                f"to {route.get('toName')} ({route.get('toCode')}) by train. "
                f"Approx duration: {route.get('approxDuration')}. "
                f"{route.get('commonInfo')} "
                f"{route.get('tourismTip')} "
                f"Train timings may change, so please check NTES before travel."
            )

            return make_train_response({
                "success": True,
                "intent": "train_route",
                "answer": answer,
                "data": route,
                "sources": [source],
            })

        from_station = get_station_by_code(from_code)
        to_station = get_station_by_code(to_code)

        if from_station and to_station:
            answer = (
                f"You can check trains from {from_station.get('name')} ({from_code}) "
                f"to {to_station.get('name')} ({to_code}). "
                f"I don't have this exact route saved yet, but you can verify live train options on NTES."
            )

            return make_train_response({
                "success": True,
                "intent": "train_route_fallback",
                "answer": answer,
                "fromStation": from_station,
                "toStation": to_station,
                "sources": [source],
            })

    # -------------------------------------------------
    # Case 6: One station info
    # -------------------------------------------------
    if len(station_codes) == 1:
        code = station_codes[0]
        station = get_station_by_code(code)

        if station:
            places = station.get("nearbyTouristPlaces", [])
            nearby_text = ", ".join(places[:4]) if places else "nearby tourist places"

            answer = (
                f"The main railway station is {station.get('name')} ({station.get('code')}) "
                f"in {station.get('city')}, {station.get('district')} district. "
                f"Nearby tourist places include {nearby_text}. "
                f"For live train timing, please check NTES."
            )

            return make_train_response({
                "success": True,
                "intent": "train_station_info",
                "answer": answer,
                "data": station,
                "sources": [source],
            })

    return make_train_response({
        "success": True,
        "intent": "train_fallback",
        "answer": (
            "I can help with Kerala train guidance. Ask like:\n"
            "- Train from Kochi to Kozhikode\n"
            "- I want to go Malappuram from Ernakulam by train\n"
            "- How to reach Munnar by train\n"
            "- Nearest railway station to Wayanad\n"
            "- What about Kerala railway?\n"
            "- List railway stations in Kerala\n\n"
            "For live timing, please check NTES."
        ),
        "sources": [source],
    })
# ============================================================
# LOCATION / TRAVEL SERVICE
# ============================================================

def clean_location_name(location: str) -> str:
    location = normalize_text(location)
    remove_words = [
        "how", "much", "time", "will", "take", "travel", "distance", "route", "drive", "go", "reach",
        "by", "car", "bike", "bus", "train", "from", "to", "near", "around",
    ]
    words = [w for w in location.split() if w not in remove_words]
    cleaned = " ".join(words).strip()

    for district in KERALA_DISTRICTS:
        if district in cleaned:
            return DISTRICT_ALIASES.get(district, district.title())

    return cleaned.title() if cleaned else location.title()


def extract_origin_destination(message: str, places: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    q = normalize_text(message)
    origin = None
    destination = None

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
            before_from = m.group(1).strip()
            after_from = m.group(2).strip()
            origin = after_from
            found = find_place_by_name(before_from, places, min_score=35)
            destination = found.get("name") if found else before_from

    if not destination:
        found = find_place_by_name(message, places, min_score=40)
        if found:
            destination = found.get("name")

    if origin:
        origin = clean_location_name(origin)

    if destination:
        destination = clean_location_name(destination)
        known = find_place_by_name(destination, places, min_score=45)
        if known:
            destination = known.get("name")

    return origin, destination


def number_to_time_from_distance_km(distance_km: float) -> str:
    if distance_km <= 0:
        return "travel time not available"

    avg_speed_kmph = 38 if distance_km > 120 else 42
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


def get_google_maps_distance(origin: str, destination: str) -> Optional[Dict[str, Any]]:
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
        res = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
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


def get_fallback_travel_estimate(origin: str, destination: str) -> Optional[Dict[str, Any]]:
    if not origin or not destination:
        return None

    o = normalize_text(origin)
    d = normalize_text(destination)

    distance_km = None
    if (o, d) in KNOWN_DISTANCES_KM:
        distance_km = KNOWN_DISTANCES_KM[(o, d)]
    elif (d, o) in KNOWN_DISTANCES_KM:
        distance_km = KNOWN_DISTANCES_KM[(d, o)]

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


def get_travel_info(message: str, places: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[str], Optional[str]]:
    origin, destination = extract_origin_destination(message, places)
    if not origin or not destination:
        return None, origin, destination

    travel_info = get_google_maps_distance(origin, destination)
    if not travel_info:
        travel_info = get_fallback_travel_estimate(origin, destination)

    return travel_info, origin, destination


# ============================================================
# PROMPTS / AI
# ============================================================

def format_place_short(place: Dict[str, Any], include_image_for_ai: bool = False) -> str:
    image_part = ""
    if include_image_for_ai:
        image_part = f", Image URL: {get_best_image_url(place) or 'No image'}"

    return (
        f"{place.get('name', 'Unknown')} "
        f"({place.get('region', 'Kerala')}) - "
        f"{safe_float(place.get('rating'), 0):.1f} star, "
        f"{safe_int(place.get('userRatings'), 0)} reviews, "
        f"Best time: {place.get('bestTime', 'Any time')}, "
        f"Distance: {place.get('distance', 'Not available')}"
        f"{image_part}, "
        f"Tags: {', '.join(place.get('tags', []))}. "
        f"Description: {clean_for_display(place.get('description', ''), 450)}"
    )


def build_system_prompt() -> str:
    return """
You are Kerala AI Guide named Octapus Ai, a smart travel assistant for the KeralaTour app.

Use the provided KeralaTour database context when available.

Developper of you is Muhammed Habeeb, place city = Kottakkal,district = Malappuram,state = Kerala,nation = India, instgram id of him = "mdhabeeb.dev".

Important rules:
- Answer as a Kerala tourism guide.
- If the user asks about a specific place, answer about the best matching database place.
- If the user explicitly asks for image/photo/pic, mention image availability and image URL only if provided in context.
- Do not mention Image URL in normal explanations.
- Never invent image URLs.
- If multiple relevant places are given, mention the best match first.
- Keep answers practical and short.
- If travel/distance info is provided, use it clearly.
- If Google Maps info is unavailable, use around/approximately.
- Do not invent live ticket prices, hotel availability, taxi booking, or reservations.
- Do not say you booked anything.
- If user asks outside Kerala tourism, politely bring back to Kerala travel.

Answer style:
- Friendly, simple English.
- Use Namaskaram 🙏 sometimes, not always.
- For place details, include highlights, best time, who it is good for, and one practical tip.
- For trip plans, use day-wise format.
- For follow-up explanations after an image, explain the same place without repeating the image URL.
""".strip()


def build_ai_context(
    message: str,
    intent: str,
    relevant_places: List[Dict[str, Any]],
    primary_place: Optional[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    moods: List[str],
    districts: List[str],
    day_count: int,
) -> str:
    include_image_for_ai = intent == "image"

    places_context = "\n".join(
        f"- {format_place_short(p, include_image_for_ai=include_image_for_ai)}"
        for p in relevant_places[:MAX_PLACES_CONTEXT]
    )

    primary_context = (
        format_place_short(primary_place, include_image_for_ai=include_image_for_ai)
        if primary_place else "No exact primary place."
    )

    travel_context = json.dumps(travel_info, indent=2) if travel_info else "No travel info available."

    return f"""
User message:
{message}

Detected intent:
{intent}

Primary matched place:
{primary_context}

Detected moods:
{", ".join(moods) if moods else "None"}

Detected districts/locations:
{", ".join(districts) if districts else "None"}

Trip day count if relevant:
{day_count}

Relevant KeralaTour database places:
{places_context if places_context else "No strong database match found."}

Travel/distance info:
{travel_context}

Now answer as Kerala AI Guide.
""".strip()


def call_groq_ai(
    user_message: str,
    intent: str,
    relevant_places: List[Dict[str, Any]],
    primary_place: Optional[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    history: List[Dict[str, str]],
    moods: List[str],
    districts: List[str],
    day_count: int,
) -> str:
    if not groq_client:
        raise RuntimeError("Groq API key not configured or ENABLE_GROQ=false")

    messages = [{"role": "system", "content": build_system_prompt()}]

    if isinstance(history, list):
        for item in history[-6:]:
            role = item.get("role")
            content = get_message_text_from_history_item(item)
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content[:1000]})

    messages.append({
        "role": "user",
        "content": build_ai_context(
            message=user_message,
            intent=intent,
            relevant_places=relevant_places,
            primary_place=primary_place,
            travel_info=travel_info,
            moods=moods,
            districts=districts,
            day_count=day_count,
        ),
    })

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=GROQ_TEMPERATURE,
        max_tokens=GROQ_MAX_TOKENS,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# LOCAL FALLBACK REPLIES
# ============================================================
def reply_for_osm_results(message: str, intent: str, osm_places: List[Dict[str, Any]]) -> str:
    if not osm_places:
        return (
            "I couldn't find matching map/service places for that query right now. "
            "Try adding a location like Kozhikode, Munnar, Kochi, Kottakkal, or Wayanad."
        )

    title = intent.replace("osm_", "").replace("_", " ").title()

    lines = [f"Here are some {title} results I found from OpenStreetMap:"]
    lines.append("")

    for i, p in enumerate(osm_places[:8], start=1):
        line = f"{i}. {p.get('name')} — {p.get('region', 'Kerala')}"

        if p.get("address"):
            line += f"\n   📍 {p.get('address')}"

        if p.get("phone"):
            line += f"\n   📞 {p.get('phone')}"

        if p.get("openingHours"):
            line += f"\n   🕒 {p.get('openingHours')}"

        lines.append(line)

    lines.append("")
    lines.append("Data source: OpenStreetMap.")

    return "\n".join(lines)


def call_groq_osm_ai(message: str, intent: str, osm_places: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
    if not groq_client:
        return reply_for_osm_results(message, intent, osm_places)

    prompt = f"""
You are Kerala AI Guide.

User asked:
{message}

Intent:
{intent}

Use only this OpenStreetMap data. Do not invent ratings, phone numbers, addresses, opening hours, or Google Maps data.

Data:
{json.dumps(osm_places[:10], ensure_ascii=False, indent=2)}

Answer rules:
- Friendly and useful
- Number the results
- Mention address/region if available
- Mention phone/opening hours only if available
- End with: Data source: OpenStreetMap.
- Do not say Google Maps
"""

    try:
        messages = [{"role": "system", "content": "You are Kerala AI Guide, a helpful Kerala travel assistant."}]

        if isinstance(history, list):
            for item in history[-4:]:
                role = item.get("role")
                content = get_message_text_from_history_item(item)
                if role in ["user", "assistant"] and content:
                    messages.append({"role": role, "content": content[:700]})

        messages.append({"role": "user", "content": prompt})

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.35,
            max_tokens=700,
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        debug_log("Groq OSM AI failed", str(e))
        return reply_for_osm_results(message, intent, osm_places)

def reply_for_travel_time(travel_info: Optional[Dict[str, Any]], origin: Optional[str], destination: Optional[str]) -> str:
    if travel_info:
        maps_line = f"\n\nRoute map: {travel_info['maps_url']}" if travel_info.get("maps_url") else ""
        source_note = "Google Maps estimate" if travel_info.get("source") == "google_maps" else "rough Kerala road estimate"
        return (
            f"Namaskaram 🙏 From {travel_info['origin']} to {travel_info['destination']}, "
            f"it is {travel_info['distance_text']} and usually takes {travel_info['duration_text']} by road.\n\n"
            f"This is a {source_note}. If it is a hill route, start early because rain, traffic, and ghat roads can slow the trip."
            f"{maps_line}"
        )

    return (
        f"Namaskaram 🙏 I understood you are asking about travel time"
        f"{f' from {origin}' if origin else ''}"
        f"{f' to {destination}' if destination else ''}. "
        "I could not get exact route data now. Try asking like: 'How much time from Malappuram to Munnar?'"
    )


def reply_for_place_details(place: Dict[str, Any]) -> str:
    return (
        f"Namaskaram 🙏 {place['name']} is a {place.get('region', 'Kerala')} destination in Kerala.\n\n"
        f"Rating: {safe_float(place.get('rating')):.1f}★ with {safe_int(place.get('userRatings'))} reviews\n"
        f"Best time: {place.get('bestTime', 'Any time')}\n"
        f"Distance: {place.get('distance') or 'Not available'}\n\n"
        f"{clean_for_display(place.get('description'), 600)}"
    )


def reply_for_recommendation(places: List[Dict[str, Any]], relevant_places: List[Dict[str, Any]], requested_count: int) -> str:
    top = relevant_places or get_trending_places(places, requested_count)
    top = top[:requested_count]

    if not top:
        return "Namaskaram 🙏 I don’t have enough places loaded from the database yet."

    lines = []
    for i, place in enumerate(top, start=1):
        image_mark = "🖼️" if has_image_url(place) else ""
        lines.append(
            f"{i}. {place['name']} {image_mark} — {place.get('region', 'Kerala')}, "
            f"{safe_float(place.get('rating')):.1f}★, best time: {place.get('bestTime', 'Any time')}"
        )

    return "Here are good Kerala places from your KeralaTour database 🌴\n\n" + "\n".join(lines)


def reply_for_compare(places: List[Dict[str, Any]]) -> str:
    if len(places) < 2:
        return "Tell me two places to compare, like: 'Munnar or Wayanad which is better?'"

    a = places[0]
    b = places[1]
    winner = a if trending_score(a) >= trending_score(b) else b

    return (
        f"Good question 🌴\n\n"
        f"{a['name']}: {a.get('region', 'Kerala')}, {safe_float(a.get('rating')):.1f}★, best time {a.get('bestTime', 'Any time')}.\n"
        f"{b['name']}: {b.get('region', 'Kerala')}, {safe_float(b.get('rating')):.1f}★, best time {b.get('bestTime', 'Any time')}.\n\n"
        f"My suggestion: choose {winner['name']} as the safer overall pick from your database rating/review strength."
    )


def reply_for_trip_plan(places: List[Dict[str, Any]], relevant_places: List[Dict[str, Any]], day_count: int) -> str:
    selected = relevant_places or get_trending_places(places, 8)
    if not selected:
        return "I need places loaded in the database to create a trip plan."

    days = []
    for day in range(1, day_count + 1):
        place = selected[(day - 1) % len(selected)]
        days.append(
            f"Day {day}: {place['name']} — explore {place.get('region', 'Kerala')}. "
            f"Best time: {place.get('bestTime', 'Any time')}. Tip: start early and keep the plan light."
        )

    return f"Here is a simple {day_count}-day Kerala trip plan based on your database 🌴\n\n" + "\n".join(days)


def local_fallback_reply(
    message: str,
    intent: str,
    places: List[Dict[str, Any]],
    relevant_places: List[Dict[str, Any]],
    primary_place: Optional[Dict[str, Any]],
    travel_info: Optional[Dict[str, Any]],
    origin: Optional[str],
    destination: Optional[str],
    day_count: int,
    requested_count: int,
) -> str:
    if intent == "image":
        return reply_for_image_request(primary_place, relevant_places)

    if intent == "travel_time":
        return reply_for_travel_time(travel_info, origin, destination)

    if intent == "compare":
        compared = find_multiple_places_in_message(message, places, limit=3)
        return reply_for_compare(compared)

    if intent == "trip_plan":
        return reply_for_trip_plan(places, relevant_places, day_count)

    if intent == "recommendation":
        return reply_for_recommendation(places, relevant_places, requested_count)

    if intent == "best_time":
        p = primary_place or (relevant_places[0] if relevant_places else None)
        if p:
            return f"For {p['name']}, the best time to visit is {p.get('bestTime', 'Any time')}.\n\n{clean_for_display(p.get('description'), 450)}"
        return "For most Kerala trips, October to March is comfortable. For monsoon beauty, June to September is beautiful, but waterfalls and hill roads need extra care."

    if primary_place:
        return reply_for_place_details(primary_place)

    if relevant_places:
        return reply_for_place_details(relevant_places[0])

    return "Namaskaram 🙏 I can help you with Kerala destinations, images, travel time, trip plans, family-safe places, beaches, backwaters, hills, wildlife, best time to visit, and route ideas."



# ============================================================
# PREMIUM KERALA LIFE AI UPGRADE MODULE
# Paste this section into your existing app.py AFTER:
# - utility functions: safe_text, safe_int, normalize_text, split_tags, pick_first_available,
#   clean_for_display, tokenize, word_overlap_score, debug_log, now_iso, stable_hash
# - Firestore db and groq_client are initialized
#
# Then add the integration lines shown in PREMIUM_UPGRADE_DOCUMENTATION.md
# ============================================================

# ============================================================
# 1) PREMIUM KERALA LIFE INTENTS
# ============================================================

KNOWLEDGE_COLLECTIONS = {
    "writer": "kerala_writers",
    "book": "kerala_books",
    "history": "kerala_history",
    "culture": "kerala_culture",
    "festival": "kerala_festivals",
    "government_service": "kerala_services",
    "food_knowledge": "kerala_foods",
    "education": "kerala_education",
    "emergency": "kerala_emergency",
    "general_kerala": "kerala_general",
}

KERALA_MASTER_INTENTS = {
    "writer": [
        "writer", "author", "poet", "novelist", "malayalam writer", "kerala writer",
        "basheer", "vaikom muhammad basheer", "vaikom basheer", "beypore sultan",
        "mt vasudevan nair", "m t vasudevan nair", "m t", "mt",
        "o v vijayan", "ov vijayan", "sugathakumari", "kumaran asan",
        "vallathol", "ulloor", "madhavikutty", "kamala das", "thakazhi",
        "s k pottekkatt", "sk pottekkatt", "kesavadev", "lalithambika antharjanam"
    ],

    "book": [
        "book", "novel", "story", "short story", "poem", "literature",
        "balyakalasakhi", "pathummayude aadu", "mathilukal", "randamoozham",
        "khasakkinte ithihasam", "chemmeen", "oru desathinte katha",
        "naalukettu", "aarachar", "mayyazhippuzhayude theerangal"
    ],

    "history": [
        "history", "kerala history", "pazhassi", "pazhassi raja", "travancore",
        "zamorin", "samoothiri", "malabar rebellion", "mappila rebellion",
        "cheraman", "king", "queen", "freedom struggle", "kunhali marakkar",
        "sakthan thampuran", "marthanda varma", "temple entry proclamation",
        "aikya kerala", "formation of kerala"
    ],

    "culture": [
        "culture", "art form", "theyyam", "kathakali", "mohiniyattam",
        "kalaripayattu", "vallam kali", "snake boat", "chenda", "oppana",
        "margam kali", "mudiyettu", "padayani", "kerala culture"
    ],

    "festival": [
        "festival", "onam", "vishu", "thrissur pooram", "pooram", "perunnal",
        "eid in kerala", "christmas in kerala", "temple festival", "boat race",
        "nehru trophy", "attukal pongala"
    ],

    "government_service": [
        "income certificate", "birth certificate", "death certificate", "ration card",
        "possession certificate", "nativity certificate", "caste certificate",
        "community certificate", "village office", "akshaya", "edistrict", "e district",
        "pension", "property tax", "building tax", "driving licence", "learner licence",
        "aadhaar", "pan card", "land tax", "encumbrance certificate", "location certificate"
    ],

    "food_knowledge": [
        "puttu", "appam", "idiyappam", "sadya", "sadhya", "biriyani",
        "malabar biriyani", "kerala food", "avial", "payasam", "porotta",
        "beef", "fish curry", "banana chips", "tapioca", "kappa", "meen curry",
        "pazhampori", "unniyappam", "ada", "kerala recipe"
    ],

    "education": [
        "course", "college", "psc", "iti", "polytechnic", "scholarship",
        "after plus two", "after 12th", "after +2", "keam", "neet",
        "higher studies", "kerala university", "ktu", "calicut university",
        "internship", "job in kerala", "career"
    ],

    "emergency": [
        "emergency", "police", "ambulance", "fire force", "helpline",
        "blood bank", "snake rescue", "disaster", "flood help", "women helpline",
        "child helpline", "accident", "urgent help", "rescue"
    ],
}

LIVE_VERIFICATION_INTENTS = {"government_service", "emergency", "education"}


def has_malayalam(text: str) -> bool:
    return bool(re.search(r"[\u0D00-\u0D7F]", safe_text(text)))


def detect_master_intent(message: str) -> str:
    """
    Premium router.
    This must run BEFORE old tourism place search.

    Priority:
    1. Location direct question
    2. Train question
    3. OSM nearby service
    4. Kerala knowledge modules
    5. Old tourism intents
    6. general_kerala
    """
    q = normalize_text(message)

    if is_location_question(message):
        return "location"

    if is_train_question(message):
        return "train"

    old_intent = detect_intent(message)

    # Nearby practical services should remain OSM
    if old_intent in OSM_TYPE_MAP:
        return old_intent

    for intent, keywords in KERALA_MASTER_INTENTS.items():
        for keyword in keywords:
            if normalize_text(keyword) in q:
                return intent

    if old_intent in [
        "image", "travel_time", "recommendation", "trip_plan",
        "best_time", "compare", "details", "family_safety", "food"
    ]:
        return old_intent

    return "general_kerala"


# ============================================================
# 2) KNOWLEDGE NORMALIZATION + SEARCH
# ============================================================

def normalize_knowledge_doc(doc_id: str, data: Dict[str, Any], intent: str) -> Dict[str, Any]:
    title = pick_first_available(
        data.get("name"),
        data.get("title"),
        fallback="Unknown"
    )

    description = pick_first_available(
        data.get("shortDescription"),
        data.get("description"),
        data.get("about"),
        data.get("summary"),
        fallback=""
    )

    tags = split_tags(data.get("tags"))
    aliases = split_tags(data.get("aliases"))
    known_for = split_tags(data.get("knownFor"))
    steps = data.get("steps", [])
    required_docs = data.get("requiredDocuments", [])

    search_blob = " ".join([
        doc_id,
        title,
        description,
        safe_text(data.get("district")),
        safe_text(data.get("region")),
        safe_text(data.get("type")),
        safe_text(data.get("category")),
        safe_text(data.get("department")),
        safe_text(data.get("platform")),
        safe_text(data.get("importance")),
        " ".join(tags),
        " ".join(aliases),
        " ".join(known_for),
        " ".join(steps) if isinstance(steps, list) else safe_text(steps),
        " ".join(required_docs) if isinstance(required_docs, list) else safe_text(required_docs),
    ])

    return {
        "id": doc_id,
        "intent": intent,
        "title": title,
        "name": title,
        "description": description,
        "type": safe_text(data.get("type") or data.get("category")),
        "district": safe_text(data.get("district")),
        "region": safe_text(data.get("region")),
        "department": safe_text(data.get("department")),
        "platform": safe_text(data.get("platform")),
        "imageUrl": safe_text(data.get("imageUrl")),
        "tags": tags,
        "aliases": aliases,
        "knownFor": known_for,
        "raw": data,
        "search_blob": normalize_text(search_blob),
    }


def knowledge_score(message: str, item: Dict[str, Any]) -> float:
    q = normalize_text(message)
    title = normalize_text(item.get("title"))
    blob = item.get("search_blob", "")
    aliases = " ".join([normalize_text(x) for x in item.get("aliases", [])])
    known_for = " ".join([normalize_text(x) for x in item.get("knownFor", [])])

    if not q:
        return 0.0

    score = 0.0

    if title and title in q:
        score += 150
    if q and q in title:
        score += 95

    for alias in item.get("aliases", []):
        a = normalize_text(alias)
        if a and a in q:
            score += 120

    for token in tokenize(q):
        if token in title:
            score += 32
        if token in aliases:
            score += 26
        if token in known_for:
            score += 24
        if token in blob:
            score += 10

    score += word_overlap_score(q, blob) * 55

    return score


def search_knowledge(message: str, intent: str, limit: int = 5) -> List[Dict[str, Any]]:
    """
    Searches the correct Firestore collection.
    Safe for MVP: reads up to 1000 docs per knowledge category.
    Later, optimize with searchKeywords array_contains.
    """
    collection = KNOWLEDGE_COLLECTIONS.get(intent)
    if not collection:
        return []

    results = []

    try:
        docs = db.collection(collection).limit(1000).stream()

        for doc in docs:
            item = normalize_knowledge_doc(doc.id, doc.to_dict() or {}, intent)
            score = knowledge_score(message, item)

            if score >= 5:
                item["_score"] = round(score, 2)
                results.append(item)

    except Exception as e:
        debug_log("Knowledge search failed", {
            "intent": intent,
            "collection": collection,
            "error": str(e),
        })
        return []

    results.sort(key=lambda x: x.get("_score", 0), reverse=True)
    return results[:limit]


# ============================================================
# 3) PREMIUM KNOWLEDGE AI PROMPT
# ============================================================

def build_knowledge_system_prompt() -> str:
    return """
You are Octopus AI, Kerala's own premium AI assistant.

Your scope:
- Kerala travel
- Kerala writers and Malayalam literature
- Kerala history
- Kerala culture and festivals
- Kerala food
- Kerala government service guidance
- Kerala education and career guidance
- Kerala emergency and nearby services
- Daily life questions useful for people from Kerala

Important rules:
- Give useful, clean, human answers.
- Use the database context when provided.
- Do not invent dates, awards, official fees, phone numbers, current rules, live status, or availability.
- For government services, mention that requirements/fees can change and users should verify through official portal/Akshaya.
- For emergency questions, be direct and safety-first.
- For literature/books, do not provide long copyrighted text or full poems/stories. Summaries and explanations are allowed.
- If the user asks in Malayalam, reply in simple Malayalam.
- If the user mixes Malayalam and English, reply in the same mixed style.
- If the question is not related to Kerala, answer briefly and connect back to Kerala if possible.
""".strip()


def build_knowledge_prompt(message: str, intent: str, results: List[Dict[str, Any]]) -> str:
    clean_results = []
    for item in results[:5]:
        raw = item.get("raw", {})
        clean_results.append({
            "id": item.get("id"),
            "title": item.get("title"),
            "type": item.get("type"),
            "district": item.get("district"),
            "region": item.get("region"),
            "description": item.get("description"),
            "knownFor": item.get("knownFor"),
            "tags": item.get("tags"),
            "raw": raw,
            "score": item.get("_score"),
        })

    language_note = "User used Malayalam. Reply in simple Malayalam." if has_malayalam(message) else "Reply in simple English."

    return f"""
User question:
{message}

Detected category:
{intent}

Language instruction:
{language_note}

Database results:
{json.dumps(clean_results, ensure_ascii=False, indent=2)}

Answer format:
- Start with a direct answer.
- Then add useful details in small sections.
- Avoid too much length.
- Add warnings/verification note only when needed.
- End with one helpful follow-up suggestion if natural.

Category-specific guidance:
- writer: include who they are, style, famous works, why important.
- book: include author, simple summary, importance, beginner note.
- history: explain simply, give period/place/importance if known.
- culture/festival: explain meaning, where seen, best time, practical note.
- government_service: give documents, steps, where to apply, verification note.
- food_knowledge: explain dish, region, what it is eaten with, simple note.
- education: give practical options, eligibility caution, Kerala context.
- emergency: give direct numbers/actions first, avoid long explanation.
""".strip()


def build_local_knowledge_reply(message: str, intent: str, results: List[Dict[str, Any]]) -> str:
    if not results:
        if intent == "emergency":
            return (
                "For urgent emergency in Kerala:\n\n"
                "- Police / general emergency: 112\n"
                "- Ambulance: 108\n"
                "- Fire and Rescue: 101\n"
                "- Women helpline: 1091\n"
                "- Child helpline: 1098\n\n"
                "If there is immediate danger, call 112 first."
            )

        return (
            "I don’t have a strong saved Kerala knowledge result for this yet. "
            "I can still answer generally, but this topic should be added to the Kerala knowledge database for a premium answer."
        )

    item = results[0]
    raw = item.get("raw", {})

    if intent == "writer":
        known_for = raw.get("knownFor", [])
        known_for_text = ", ".join(known_for[:6]) if isinstance(known_for, list) else safe_text(known_for)
        importance = safe_text(raw.get("importance"), "Details not added yet.")

        return (
            f"{item['title']} is an important Malayalam writer from Kerala.\n\n"
            f"{item.get('description')}\n\n"
            f"Famous works: {known_for_text or 'Not added yet'}\n\n"
            f"Why important: {importance}"
        )

    if intent == "government_service":
        steps = raw.get("steps", [])
        docs = raw.get("requiredDocuments", [])

        steps_text = "\n".join([f"{i+1}. {s}" for i, s in enumerate(steps)]) if isinstance(steps, list) else safe_text(steps)
        docs_text = "\n".join([f"- {d}" for d in docs]) if isinstance(docs, list) else safe_text(docs)

        return (
            f"{item['title']}\n\n"
            f"{item.get('description')}\n\n"
            f"Documents usually needed:\n{docs_text or '- Not added yet'}\n\n"
            f"Basic steps:\n{steps_text or '- Not added yet'}\n\n"
            "Note: Government rules, documents, and fees can change. Please verify through the official portal or Akshaya."
        )

    if intent == "emergency":
        return (
            f"{item['title']}\n\n"
            f"{item.get('description')}\n\n"
            "If there is immediate danger, call 112 first."
        )

    return (
        f"{item['title']}\n\n"
        f"{item.get('description')}\n\n"
        "Ask me for a simpler explanation, Malayalam explanation, or related topics."
    )


def call_groq_knowledge_ai(message: str, intent: str, results: List[Dict[str, Any]], history: List[Dict[str, str]]) -> str:
    if not groq_client:
        return build_local_knowledge_reply(message, intent, results)

    messages = [{"role": "system", "content": build_knowledge_system_prompt()}]

    if isinstance(history, list):
        for item in history[-5:]:
            role = item.get("role")
            content = get_message_text_from_history_item(item)
            if role in ["user", "assistant"] and content:
                messages.append({"role": role, "content": content[:800]})

    messages.append({
        "role": "user",
        "content": build_knowledge_prompt(message, intent, results)
    })

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.32,
        max_tokens=850,
    )

    return response.choices[0].message.content.strip()


# ============================================================
# 4) PREMIUM RESPONSE STANDARD
# ============================================================

def get_response_type_for_knowledge(intent: str) -> str:
    if intent in ["writer", "book", "history", "culture", "festival", "food_knowledge", "general_kerala"]:
        return "knowledge_card"

    if intent == "government_service":
        return "service_guide"

    if intent == "education":
        return "education_guide"

    if intent == "emergency":
        return "emergency_card"

    return "knowledge_card"


def build_suggested_questions(intent: str, primary: Optional[Dict[str, Any]]) -> List[str]:
    name = safe_text(primary.get("title")) if primary else "this topic"

    if intent == "writer":
        return [
            f"Famous works of {name}",
            f"Why is {name} important?",
            f"Explain {name} in Malayalam",
        ]

    if intent == "book":
        return [
            f"Explain {name} simply",
            f"Who wrote {name}?",
            "Suggest Malayalam books for beginners",
        ]

    if intent == "government_service":
        return [
            "What documents are needed?",
            "Where can I apply?",
            "How long will it take?",
        ]

    if intent == "history":
        return [
            f"Explain {name} simply",
            "Show important Kerala history events",
            "Tell this like a story",
        ]

    if intent == "emergency":
        return [
            "Show emergency numbers",
            "Nearest hospital near me",
            "Police station near me",
        ]

    if intent == "food_knowledge":
        return [
            f"How to make {name}?",
            "Famous Kerala breakfast items",
            "Best Malabar food items",
        ]

    return [
        "Explain simply",
        "Give more details",
        "Tell in Malayalam",
    ]


def build_knowledge_card(item: Dict[str, Any]) -> Dict[str, Any]:
    raw = item.get("raw", {})
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "name": item.get("title"),
        "intent": item.get("intent"),
        "type": item.get("type"),
        "district": item.get("district"),
        "region": item.get("region"),
        "description": clean_for_display(item.get("description"), 450),
        "imageUrl": item.get("imageUrl") or None,
        "tags": item.get("tags", []),
        "knownFor": item.get("knownFor", []),
        "score": item.get("_score", 0),
        "source": "Kerala Knowledge Database",
        "extra": {
            "born": raw.get("born"),
            "died": raw.get("died"),
            "department": raw.get("department"),
            "platform": raw.get("platform"),
            "importance": raw.get("importance"),
            "requiredDocuments": raw.get("requiredDocuments"),
            "steps": raw.get("steps"),
            "officialLink": raw.get("officialLink"),
        },
    }


def handle_kerala_knowledge_question(
    message: str,
    intent: str,
    history: List[Dict[str, str]]
) -> Dict[str, Any]:
    results = search_knowledge(message, intent, limit=5)

    # fallback: if no result in exact category, search general_kerala
    if not results and intent != "general_kerala":
        general_results = search_knowledge(message, "general_kerala", limit=3)
        results = general_results

    primary = results[0] if results else None

    try:
        reply = call_groq_knowledge_ai(message, intent, results, history)
        ai_used = bool(groq_client)
    except Exception as e:
        debug_log("Knowledge AI failed", str(e))
        reply = build_local_knowledge_reply(message, intent, results)
        ai_used = False

    response_type = get_response_type_for_knowledge(intent)

    return {
        "success": True,
        "reply": reply,
        "answer": reply,
        "intent": intent,
        "masterIntent": intent,
        "responseType": response_type,
        "aiUsed": ai_used,
        "confidence": primary.get("_score", 0) if primary else 0,

        "answerQuality": {
            "source": "database" if primary else "ai_general_or_fallback",
            "needsLiveVerification": intent in LIVE_VERIFICATION_INTENTS,
            "canAnswer": bool(reply),
            "hasDatabaseResult": bool(primary),
        },

        "primaryResult": build_knowledge_card(primary) if primary else None,
        "results": [build_knowledge_card(x) for x in results],

        # compatibility with your existing frontend names
        "primaryPlace": None,
        "matchedPlaces": [],

        "ui": {
            "showCards": bool(results),
            "cardMode": response_type,
            "showImages": False,
            "showBigImage": False,
            "showMap": False,
            "showActions": True,
            "cardLimit": min(len(results), 5),
            "reason": "premium_kerala_knowledge",
        },

        "suggestedQuestions": build_suggested_questions(intent, primary),
        "timestamp": now_iso(),
        "requestId": stable_hash(message + str(time.time())),
        "source": "Kerala Knowledge Database",
    }


# ============================================================
# 5) RESPONSE TYPE FOR OLD TOURISM RESULTS
# Add this to old response if you want cleaner frontend routing.
# ============================================================

def get_response_type(intent: str, ui: Dict[str, Any]) -> str:
    if intent == "image":
        return "image"

    if intent == "trip_plan":
        return "trip_plan"

    if intent == "live_search":
        return "live_search"

    if intent in OSM_TYPE_MAP:
        return "service_cards"

    if ui.get("showCards"):
        return "place_cards"

    return "text"


# ============================================================
# 6) MAP ROUTE FOR MULTIPLE PLACE CARDS
# Add "mapRoute": build_map_route(cards) to your existing tourism return.
# ============================================================

def build_map_route(places: List[Dict[str, Any]]) -> Dict[str, Any]:
    points = []

    for p in places:
        lat = safe_float(p.get("lat") or p.get("latitude"), 0)
        lng = safe_float(p.get("lng") or p.get("longitude") or p.get("lon"), 0)

        if lat and lng:
            points.append({
                "id": p.get("id"),
                "name": p.get("name"),
                "lat": lat,
                "lng": lng,
            })

    return {
        "enabled": len(points) >= 2,
        "points": points,
    }


# ============================================================
# 7) BUDGET + PEOPLE EXTRACTION FOR SMART TRIP PLANNER
# ============================================================

def extract_budget(message: str) -> Optional[int]:
    q = normalize_text(message)
    patterns = [
        r"under\s*₹?\s*(\d+)",
        r"below\s*₹?\s*(\d+)",
        r"budget\s*₹?\s*(\d+)",
        r"within\s*₹?\s*(\d+)",
        r"under\s*(\d+)\s*rs",
        r"under\s*(\d+)\s*rupees",
        r"below\s*(\d+)\s*rs",
        r"(\d+)\s*budget",
    ]

    for p in patterns:
        m = re.search(p, q)
        if m:
            return int(m.group(1))

    return None


def extract_people_count(message: str) -> int:
    q = normalize_text(message)

    m = re.search(r"(\d+)\s*(people|persons|friends|members|family|students)", q)
    if m:
        return max(1, min(30, int(m.group(1))))

    if "couple" in q or "honeymoon" in q:
        return 2

    return 1


# ============================================================
# CHAT ORCHESTRATION
# ============================================================

def choose_primary_place(message: str, intent: str, places: List[Dict[str, Any]], relevant_places: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if relevant_places:
        best = relevant_places[0]
        if safe_float(best.get("_score"), 0) >= 30:
            return best

    min_score = 30 if intent == "image" else 45
    return find_place_by_name(message, places, min_score=min_score)

def is_location_question(message: str) -> bool:
    msg = (message or "").lower().strip()

    direct_location_questions = [
        "where am i",
        "where i am",
        "current location",
        "my current location",
        "my location",
        "where is my location",
    ]

    service_words = [
        "hotel", "stay", "room", "resort", "lodge", "homestay",
        "restaurant", "food", "cafe", "hospital", "pharmacy",
        "atm", "bank", "fuel", "petrol", "station",
        "bus", "train", "airport", "tonight"
    ]

    has_direct_location_question = any(k in msg for k in direct_location_questions)
    has_service_need = any(k in msg for k in service_words)

    return has_direct_location_question and not has_service_need
def build_chat_result(
    message: str,
    history: List[Dict[str, str]],
    current_place_id: Optional[str] = None,
    last_matched_place_ids: Optional[List[str]] = None,
    user_lat: float = 0.0,
    user_lng: float = 0.0,
    user_location_text: str = "",
) -> Dict[str, Any]:
    places = load_places_from_firestore()

    # ✅ Handle current location questions directly.
    # Do this before search_places(), OSM, or Groq, otherwise AI may guess wrong.
    if is_location_question(message):
        clean_location = user_location_text.strip() if user_location_text else ""

        if clean_location:
            return {
                "reply": f"You are currently around {clean_location}.",
                "intent": "location",
                "aiUsed": False,
                "ui": {
                    "showCards": False,
                    "showBigImage": False,
                    "showImages": False,
                    "cardMode": "none",
                    "cardLimit": 0,
                },
                "currentPlaceId": current_place_id,
                "lastMatchedPlaceIds": last_matched_place_ids or [],
                "selectedIndex": None,
                "isFollowup": False,
                "moods": [],
                "districts": [],
                "dayCount": 0,
                "requestedCount": 0,
                "origin": None,
                "destination": None,
                "primaryPlace": None,
                "media": None,
                "matchedPlaces": [],
                "travelInfo": None,
                "placeCount": len(places),
                "imageCount": len([p for p in places if has_image_url(p)]),
                "timestamp": now_iso(),
                "requestId": stable_hash(message + str(time.time())),
                "source": "Browser Location",
                "userLocation": {
                    "lat": user_lat,
                    "lng": user_lng,
                    "text": clean_location,
                },
            }

        if user_lat and user_lng:
            return {
                "reply": f"I detected your coordinates as {user_lat}, {user_lng}, but I could not convert them into a readable place name.",
                "intent": "location",
                "aiUsed": False,
                "ui": {
                    "showCards": False,
                    "showBigImage": False,
                    "showImages": False,
                    "cardMode": "none",
                    "cardLimit": 0,
                },
                "currentPlaceId": current_place_id,
                "lastMatchedPlaceIds": last_matched_place_ids or [],
                "selectedIndex": None,
                "isFollowup": False,
                "moods": [],
                "districts": [],
                "dayCount": 0,
                "requestedCount": 0,
                "origin": None,
                "destination": None,
                "primaryPlace": None,
                "media": None,
                "matchedPlaces": [],
                "travelInfo": None,
                "placeCount": len(places),
                "imageCount": len([p for p in places if has_image_url(p)]),
                "timestamp": now_iso(),
                "requestId": stable_hash(message + str(time.time())),
                "source": "Browser Coordinates",
                "userLocation": {
                    "lat": user_lat,
                    "lng": user_lng,
                    "text": "",
                },
            }

        return {
            "reply": "I could not access your current location. Please allow location permission in your browser.",
            "intent": "location",
            "aiUsed": False,
            "ui": {
                "showCards": False,
                "showBigImage": False,
                "showImages": False,
                "cardMode": "none",
                "cardLimit": 0,
            },
            "currentPlaceId": current_place_id,
            "lastMatchedPlaceIds": last_matched_place_ids or [],
            "selectedIndex": None,
            "isFollowup": False,
            "moods": [],
            "districts": [],
            "dayCount": 0,
            "requestedCount": 0,
            "origin": None,
            "destination": None,
            "primaryPlace": None,
            "media": None,
            "matchedPlaces": [],
            "travelInfo": None,
            "placeCount": len(places),
            "imageCount": len([p for p in places if has_image_url(p)]),
            "timestamp": now_iso(),
            "requestId": stable_hash(message + str(time.time())),
            "source": "Location Permission",
            "userLocation": {
                "lat": user_lat,
                "lng": user_lng,
                "text": "",
            },
        }

    intent = detect_intent(message)
    moods = extract_moods(message)
    districts = extract_districts(message)
    day_count = extract_day_count(message)
    requested_count = extract_requested_count(message, default=DEFAULT_CARD_LIMIT)
    selected_index = detect_selected_index(message)
    followup = is_followup_message(message)
    is_osm_service_intent = intent in OSM_TYPE_MAP and (intent != "food" or is_food_place_query(message))


    # ============================================================
    # V3 LIVE SEARCH
    # ============================================================

    if should_use_live_search(message) and not is_osm_service_intent:
        live_query = build_live_search_query(message)
        live_data = tavily_live_search(live_query, max_results=3)

        debug_log("V3 live search triggered", {
            "message": message,
            "query": live_query,
            "ok": live_data.get("ok"),
            "error": live_data.get("error"),
        })

        if live_data.get("ok"):
            live_answer = live_data.get("answer") or ""
            live_results = live_data.get("results", [])

            reply = live_answer.strip()

            if not reply:
                reply = "I found some live search results for your question."

            return {
                "reply": reply,
                "intent": "live_search",
                "aiUsed": False,
                "ui": {
                    "showCards": False,
                    "showBigImage": False,
                    "showImages": False,
                    "cardMode": "live_search",
                    "cardLimit": 0,
                },
                "currentPlaceId": current_place_id,
                "lastMatchedPlaceIds": last_matched_place_ids or [],
                "selectedIndex": None,
                "isFollowup": False,
                "moods": moods,
                "districts": districts,
                "dayCount": day_count,
                "requestedCount": requested_count,
                "origin": None,
                "destination": None,
                "primaryPlace": None,
                "media": None,
                "matchedPlaces": [],
                "travelInfo": None,
                "liveAnswer": live_answer,
                "liveResults": live_results,
                "placeCount": len(places),
                "imageCount": len([p for p in places if has_image_url(p)]),
                "timestamp": now_iso(),
                "requestId": stable_hash(message + str(time.time())),
                "source": "Tavily Live Search",
            }

    current_place = get_place_by_id(current_place_id)
    history_place = extract_context_place_from_history(history, places)
    selected_from_list = select_place_from_previous_list(selected_index, last_matched_place_ids or [], current_place, places)

    relevant_places = search_places(
        query=message,
        places=places,
        limit=max(MAX_PLACES_CONTEXT, requested_count),
        min_score=3 if intent == "image" else 5,
        require_image=False,
    )
    user_location_context = build_user_location_context(
        user_lat=user_lat,
        user_lng=user_lng,
        user_location_text=user_location_text,
    )

    osm_search_message = message

    if is_near_me_query(message) and user_location_context:
        osm_search_message = f"{message} near {user_location_context}"

    osm_places = search_osm_places(
        message=osm_search_message,
        intent=intent,
        limit=min(requested_count, MAX_MATCHED_PLACES_RESPONSE),
        user_lat=user_lat,
        user_lng=user_lng,
    )   

    # OSM/service data route
        # OSM/service data route
    if intent in OSM_TYPE_MAP and (intent != "food" or is_food_place_query(message)):
        requested_count = extract_requested_count(message, default=DEFAULT_CARD_LIMIT)

        

        reply = call_groq_osm_ai(
            message=(
                f"{message}\n\n"
                f"User current location: {user_location_text or user_location_context or 'not available'}"
            ),
            intent=intent,
            osm_places=osm_places,
            history=history if isinstance(history, list) else [],
        )

        ui = build_ui_directives(
            intent=intent,
            message=message,
            primary_place=osm_places[0] if osm_places else None,
            matched_count=len(osm_places),
            is_followup=False,
            requested_count=requested_count,
        )

        return {
            "reply": reply,
            "intent": intent,
            "aiUsed": bool(groq_client),
            "ui": ui,
            "currentPlaceId": osm_places[0].get("id") if osm_places else current_place_id,
            "lastMatchedPlaceIds": [p.get("id") for p in osm_places if p.get("id")],
            "selectedIndex": None,
            "isFollowup": False,
            "moods": [],
            "districts": [],
            "dayCount": 0,
            "requestedCount": requested_count,
            "origin": None,
            "destination": None,
            "primaryPlace": build_place_card(osm_places[0], include_description=True, include_image=False) if osm_places else None,
            "media": None,
            "matchedPlaces": [
                build_place_card(p, include_description=True, include_image=False)
                for p in osm_places[:safe_int(ui.get("cardLimit"), DEFAULT_CARD_LIMIT)]
            ],
            "travelInfo": None,
            "placeCount": len(places),
            "imageCount": len([p for p in places if has_image_url(p)]),
            "timestamp": now_iso(),
            "requestId": stable_hash(message + str(time.time())),
            "source": "OpenStreetMap",
            "userLocation": {
                "lat": user_lat,
                "lng": user_lng,
                "text": user_location_text,
            },
        }
    if intent == "image":
        image_matches = search_places(
            message,
            places,
            limit=max(MAX_PLACES_CONTEXT, requested_count),
            min_score=3,
            require_image=True,
        )
        if image_matches:
            relevant_places = image_matches

    if intent in ["recommendation", "general"] and not relevant_places:
        relevant_places = get_trending_places(places, limit=max(MAX_PLACES_CONTEXT, requested_count))

    if intent == "compare":
        relevant_places = find_multiple_places_in_message(message, places, limit=MAX_PLACES_CONTEXT)

    if intent == "trip_plan" and not relevant_places:
        relevant_places = get_trending_places(places, limit=MAX_PLACES_CONTEXT)

    primary_place = choose_primary_place(message, intent, places, relevant_places)

    
    # Practical follow-up behavior:
    # "pic one" should use selected item from previous list.
    # "explain it" should use currentPlaceId / previous selected place.
    if followup:
        followup_place = selected_from_list or current_place or history_place or primary_place
        if followup_place:
            primary_place = followup_place
            if intent == "image":
                relevant_places = [followup_place]
            elif intent not in ["recommendation", "trip_plan", "compare"]:
                relevant_places = [followup_place]

    travel_info = None
    origin = None
    destination = None

    if intent == "travel_time":
        travel_info, origin, destination = get_travel_info(message, places)
        if destination:
            destination_place = find_place_by_name(destination, places, min_score=35)
            if destination_place:
                primary_place = destination_place
                existing_ids = {p["id"] for p in relevant_places}
                if destination_place["id"] not in existing_ids:
                    relevant_places.insert(0, destination_place)

    ui = build_ui_directives(
        intent=intent,
        message=message,
        primary_place=primary_place,
        matched_count=len(relevant_places),
        is_followup=followup,
        requested_count=requested_count,
    )

    media = None
    if intent == "image" and ui.get("showBigImage"):
        media = build_media_response(primary_place, relevant_places)

    ai_used = False

    try:
        if intent == "image":
            reply = reply_for_image_request(primary_place, relevant_places)
        else:
            reply = call_groq_ai(
                user_message=message,
                intent=intent,
                relevant_places=relevant_places,
                primary_place=primary_place,
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
            primary_place=primary_place,
            travel_info=travel_info,
            origin=origin,
            destination=destination,
            day_count=day_count,
            requested_count=requested_count,
        )

    card_limit = safe_int(ui.get("cardLimit"), MAX_MATCHED_PLACES_RESPONSE)
    card_limit = max(0, min(card_limit, MAX_MATCHED_PLACES_RESPONSE))
    cards = relevant_places[:card_limit]

    response_last_ids = [p.get("id") for p in cards if p.get("id")]

    return {
        "reply": reply,
        "intent": intent,
        "masterIntent": intent,
        "responseType": get_response_type(intent, ui),
        "aiUsed": ai_used,
        "ui": ui,
        "currentPlaceId": primary_place.get("id") if primary_place else current_place_id,
        "lastMatchedPlaceIds": response_last_ids,
        "selectedIndex": selected_index,
        "isFollowup": followup,
        "moods": moods,
        "districts": districts,
        "dayCount": day_count,
        "requestedCount": requested_count,
        "origin": origin,
        "destination": destination,
        "primaryPlace": build_place_card(
            primary_place,
            include_description=True,
            include_image=bool(ui.get("showImages")),
        ) if primary_place else None,
        "media": media,
        "matchedPlaces": [
            build_place_card(
                p,
                include_description=True,
                include_image=True,
            )
            for p in cards
        ],
        "travelInfo": travel_info,
        "mapRoute": build_map_route(cards),
        "budget": extract_budget(message),
        "peopleCount": extract_people_count(message),
        "placeCount": len(places),
        "imageCount": len([p for p in places if has_image_url(p)]),
        "timestamp": now_iso(),
        "requestId": stable_hash(message + str(time.time())),
    }


# ============================================================
# API ROUTES
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "ok",
        "name": APP_NAME,
        "env": ENV,
        "groqConfigured": bool(GROQ_API_KEY and ENABLE_GROQ),
        "googleMapsConfigured": bool(GOOGLE_MAPS_API_KEY),
        "firestoreCollection": FIRESTORE_COLLECTION,
        "routes": [
            "POST /api/chat",
            "GET /api/places",
            "GET /api/search?q=munnar",
            "GET /api/images?q=munnar",
            "GET /api/image?q=munnar",
            "GET /api/trending",
            "GET /api/place/<id>",
            "GET|POST /api/cache/refresh",
            "GET /api/health",
            "GET /api/osm/search?q=hospital kozhikode",
            "GET /api/osm/categories",
            "POST /api/debug/intent",
        ],
    })


@app.route("/api/health", methods=["GET"])
def health_api():
    cache_age = round(time.time() - PLACES_CACHE_TIME, 2) if PLACES_CACHE_TIME else None

    osm_sample_available = False
    try:
        osm_sample_available = len(list(db.collection(OSM_COLLECTION).limit(1).stream())) > 0
    except Exception:
        osm_sample_available = False

    return jsonify({
        "status": "ok",
        "name": APP_NAME,
        "time": now_iso(),
        "cacheLoaded": bool(PLACES_CACHE),
        "cacheAgeSeconds": cache_age,
        "placeCount": len(PLACES_CACHE),
        "imageCount": len([p for p in PLACES_CACHE if has_image_url(p)]),
        "groqConfigured": bool(groq_client),
        "googleMapsConfigured": bool(GOOGLE_MAPS_API_KEY),
        "osmCollection": OSM_COLLECTION,
        "osmSampleAvailable": osm_sample_available,
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        body = request.get_json(force=True) or {}

        message = safe_text(body.get("message"))
        history = body.get("history", [])
        current_place_id = safe_text(body.get("currentPlaceId") or body.get("placeId"))
        last_matched_place_ids = body.get("lastMatchedPlaceIds", [])

        user_lat = safe_float(body.get("userLat"), 0.0)
        user_lng = safe_float(body.get("userLng"), 0.0)
        user_location_text = safe_text(body.get("userLocationText"))

        if not isinstance(last_matched_place_ids, list):
            last_matched_place_ids = []

        if not message:
            return jsonify({
                "reply": "Please ask me anything about Kerala — travel, writers, history, culture, services, food, emergency, or daily life.",
                "error": "empty_message"
            }), 400

        # ============================================================
        # PREMIUM MASTER ROUTER
        # Routes Kerala life/knowledge questions before tourism search.
        # ============================================================

        master_intent = detect_master_intent(message)

        if master_intent == "train":
            train_result = handle_train_question(message)
            train_result["masterIntent"] = "train"
            train_result["responseType"] = "train_text"
            return jsonify(train_result)

        if master_intent in [
            "writer", "book", "history", "culture", "festival",
            "government_service", "food_knowledge", "education",
            "emergency", "general_kerala"
        ]:
            result = handle_kerala_knowledge_question(
                message=message,
                intent=master_intent,
                history=history if isinstance(history, list) else []
            )
            return jsonify(result)

        result = build_chat_result(
            message=message,
            history=history,
            current_place_id=current_place_id,
            last_matched_place_ids=last_matched_place_ids,
            user_lat=user_lat,
            user_lng=user_lng,
            user_location_text=user_location_text,
        )

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
        limit = safe_int(request.args.get("limit"), 100)
        limit = max(1, min(limit, 5000))
        places = load_places_from_firestore()
        return jsonify({
            "count": len(places),
            "places": places[:limit],
            "cacheAgeSeconds": round(time.time() - PLACES_CACHE_TIME, 2) if PLACES_CACHE_TIME else None,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def search_api():
    try:
        q = safe_text(request.args.get("q"))
        limit = safe_int(request.args.get("limit"), 10)
        include_osm = safe_text(request.args.get("includeOsm")).lower() in ["1", "true", "yes"]
        limit = max(1, min(limit, 100))

        if not q:
            return jsonify({"query": q, "count": 0, "places": []})

        places = load_places_from_firestore()
        tourism_results = search_places(q, places, limit=limit, min_score=3)

        results = [
            build_place_card(p, include_description=True, include_image=True)
            for p in tourism_results
        ]

        detected = detect_intent(q)

        if include_osm or detected in OSM_TYPE_MAP or is_food_place_query(q):
            osm_intent = detected if detected in OSM_TYPE_MAP else "food"
            osm_results = search_osm_places(q, intent=osm_intent, limit=limit)
            results.extend([
                build_place_card(p, include_description=True, include_image=True)
                for p in osm_results
            ])

        return jsonify({
            "query": q,
            "count": len(results[:limit]),
            "tourismCount": len(tourism_results),
            "osmCount": len([p for p in results if p.get("isOsm")]),
            "places": results[:limit],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/images", methods=["GET"])
def images_api():
    try:
        q = safe_text(request.args.get("q"))
        limit = safe_int(request.args.get("limit"), 20)
        limit = max(1, min(limit, 100))
        places = load_places_from_firestore()

        if q:
            results = search_places(q, places, limit=limit, min_score=3, require_image=True)
        else:
            results = get_trending_places(places, limit=limit, require_image=True)

        return jsonify({
            "query": q,
            "count": len(results),
            "images": [
                {
                    "placeId": p.get("id"),
                    "name": p.get("name"),
                    "region": p.get("region"),
                    "imageUrl": get_best_image_url(p),
                    "rating": p.get("rating"),
                    "score": p.get("_score"),
                }
                for p in results
            ],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/image", methods=["GET"])
def single_image_api():
    try:
        q = safe_text(request.args.get("q"))
        if not q:
            return jsonify({"error": "missing_query", "message": "Use /api/image?q=munnar"}), 400

        places = load_places_from_firestore()
        matches = search_places(q, places, limit=8, min_score=3, require_image=False)
        image_matches = [p for p in matches if has_image_url(p)]
        primary = image_matches[0] if image_matches else (matches[0] if matches else find_place_by_name(q, places, min_score=30))
        media = build_media_response(primary, matches)

        return jsonify({
            "query": q,
            "media": media,
            "matchedPlaces": [build_place_card(p, include_description=False, include_image=True) for p in matches],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trending", methods=["GET"])
def trending_api():
    try:
        limit = safe_int(request.args.get("limit"), 20)
        limit = max(1, min(limit, 100))
        images_only = safe_text(request.args.get("imagesOnly")).lower() in ["1", "true", "yes"]
        places = load_places_from_firestore()
        results = get_trending_places(places, limit=limit, require_image=images_only)

        return jsonify({
            "count": len(results),
            "places": [build_place_card(p, include_description=True, include_image=True) for p in results],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/v3/live-search", methods=["GET"])
def api_v3_live_search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "ok": False,
            "error": "Missing query. Use /api/v3/live-search?q=your search"
        }), 400

    search_query = build_live_search_query(query)
    result = tavily_live_search(search_query, max_results=3)

    return jsonify(result)

@app.route("/api/place/<place_id>", methods=["GET"])
def place_detail_api(place_id: str):
    try:
        place = get_place_by_id(place_id)
        if not place:
            return jsonify({"error": "place_not_found", "placeId": place_id}), 404
        return jsonify({"place": build_place_card(place, include_description=True, include_image=True), "raw": place})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/osm/search", methods=["GET"])
def osm_search_api():
    try:
        q = safe_text(request.args.get("q"))
        intent = safe_text(request.args.get("intent"))
        limit = safe_int(request.args.get("limit"), 10)
        limit = max(1, min(limit, 50))

        if not q:
            return jsonify({
                "query": q,
                "intent": intent,
                "count": 0,
                "places": [],
            })

        if not intent:
            intent = detect_intent(q)

        if intent == "food" and not is_food_place_query(q):
            intent = "food"

        if intent not in OSM_TYPE_MAP:
            # Try food as fallback for restaurants/cafes
            if is_food_place_query(q):
                intent = "food"
            else:
                intent = "osm_health"

        results = search_osm_places(q, intent=intent, limit=limit)

        return jsonify({
            "query": q,
            "intent": intent,
            "count": len(results),
            "places": [
                build_place_card(p, include_description=True, include_image=True)
                for p in results
            ],
            "source": "OpenStreetMap",
        })

    except Exception as e:
        debug_log("Server error in /api/osm/search", {"error": str(e), "trace": traceback.format_exc()})
        return jsonify({"error": str(e)}), 500


@app.route("/api/osm/categories", methods=["GET"])
def osm_categories_api():
    return jsonify({
        "collection": OSM_COLLECTION,
        "categories": OSM_TYPE_MAP,
        "examples": [
            "/api/osm/search?q=hospital kozhikode",
            "/api/osm/search?q=restaurant munnar",
            "/api/osm/search?q=airport kerala",
            "/api/osm/search?q=pharmacy kottakkal",
            "/api/osm/search?q=railway station kochi",
        ],
    })
@app.route("/api/cache/refresh", methods=["POST", "GET"])
def refresh_cache_api():
    try:
        places = load_places_from_firestore(force=True)
        return jsonify({
            "status": "ok",
            "count": len(places),
            "imageCount": len([p for p in places if has_image_url(p)]),
            "cacheTime": PLACES_CACHE_TIME,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/debug/intent", methods=["POST"])
def debug_intent_api():
    body = request.get_json(force=True) or {}
    message = safe_text(body.get("message"))
    history = body.get("history", [])
    current_place_id = safe_text(body.get("currentPlaceId") or body.get("placeId"))
    last_matched_place_ids = body.get("lastMatchedPlaceIds", [])

    places = load_places_from_firestore()
    intent = detect_intent(message)
    master_intent = detect_master_intent(message)
    matches = search_places(message, places, limit=8, min_score=3)
    primary = choose_primary_place(message, intent, places, matches)

    selected_index = detect_selected_index(message)
    current_place = get_place_by_id(current_place_id)
    selected_from_list = select_place_from_previous_list(selected_index, last_matched_place_ids if isinstance(last_matched_place_ids, list) else [], current_place, places)

    travel_info, origin, destination = get_travel_info(message, places)

    return jsonify({
        "message": message,
        "intent": intent,
        "masterIntent": master_intent,
        "isImageRequest": is_image_request(message),
        "isExplicitImageRequest": is_explicit_image_request(message),
        "isFollowup": is_followup_message(message),
        "selectedIndex": selected_index,
        "moods": extract_moods(message),
        "districts": extract_districts(message),
        "dayCount": extract_day_count(message),
        "requestedCount": extract_requested_count(message),
        "origin": origin,
        "destination": destination,
        "currentPlace": build_place_card(current_place, include_description=True, include_image=True) if current_place else None,
        "selectedFromList": build_place_card(selected_from_list, include_description=True, include_image=True) if selected_from_list else None,
        "primaryPlace": build_place_card(primary, include_description=True, include_image=True) if primary else None,
        "travelInfo": travel_info,
        "matches": [build_place_card(p, include_description=True, include_image=True) for p in matches],
    })


# ============================================================
# FRONTEND RULE
# ============================================================
# Store these after every response:
# currentPlaceId = data.currentPlaceId
# lastMatchedPlaceIds = data.lastMatchedPlaceIds
#
# Send them in the next request:
# {
#   message,
#   history,
#   currentPlaceId,
#   lastMatchedPlaceIds
# }
#
# Render rule:
# if data.ui.showBigImage && data.media?.status === "found": show big image
# if data.ui.showCards: show matchedPlaces cards
# if data.ui.showImages is false: do not render imageUrl inside cards
# ============================================================


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print(APP_NAME)
    print("=" * 60)
    print(f"ENV: {ENV}")
    print(f"Groq configured: {bool(groq_client)}")
    print(f"Google Maps configured: {bool(GOOGLE_MAPS_API_KEY)}")
    print(f"Firebase service account file: {FIREBASE_SERVICE_ACCOUNT}")
    print(f"Firebase JSON env configured: {bool(FIREBASE_SERVICE_ACCOUNT_JSON)}")
    print(f"Firestore collection: {FIRESTORE_COLLECTION}")
    print(f"Model: {GROQ_MODEL}")
    print(f"Port: {PORT}")
    print("=" * 60)

    debug_mode = ENV == "development"
    app.run(host="0.0.0.0", port=PORT, debug=debug_mode)
