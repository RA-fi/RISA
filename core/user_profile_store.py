"""
Per-device user profile store for RISA.
Passively accumulates personal facts from conversations — never asks the user directly.
Facts are extracted from what users naturally share and used to personalize advice.
"""

import json
import os
import re
import sqlite3
import threading
import time
from typing import Dict, List, Optional

# core/ lives one level below project root; chroma_db/ is at project root
_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "chroma_db", "sessions.db")
_lock = threading.Lock()


def _get_conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_profile_table() -> None:
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                device_key   TEXT PRIMARY KEY,
                name         TEXT,
                age          INTEGER,
                gender       TEXT,
                weight_kg    REAL,
                blood_group  TEXT,
                conditions   TEXT,
                allergies    TEXT,
                medications  TEXT,
                location     TEXT,
                updated_at   INTEGER NOT NULL DEFAULT 0
            );
        """)
        conn.commit()
        conn.close()


def get_profile(device_key: str) -> dict:
    """Return the stored profile dict for this device, or {}."""
    if not device_key:
        return {}
    with _lock:
        conn = _get_conn()
        try:
            row = conn.execute(
                "SELECT * FROM user_profiles WHERE device_key = ?", (device_key,)
            ).fetchone()
        except Exception:
            row = None
        finally:
            conn.close()
    if not row:
        return {}
    profile = dict(row)
    for field in ("conditions", "allergies", "medications"):
        raw = profile.get(field)
        if raw:
            try:
                profile[field] = json.loads(raw)
            except Exception:
                profile[field] = [raw]
        else:
            profile[field] = []
    return profile


def merge_profile(device_key: str, updates: dict) -> None:
    """
    Merge newly extracted facts into the existing profile.
    List fields (conditions, allergies, medications) are unioned.
    Scalar fields are updated only if not already known, except 'age' which always updates.
    """
    if not device_key or not updates:
        return
    existing = get_profile(device_key)
    merged = dict(existing)

    for key, val in updates.items():
        if val is None:
            continue
        if key in ("conditions", "allergies", "medications"):
            existing_list = existing.get(key) or []
            new_items = val if isinstance(val, list) else [val]
            seen = {v.lower() for v in existing_list}
            combined = list(existing_list)
            for item in new_items:
                if item.lower() not in seen:
                    combined.append(item)
                    seen.add(item.lower())
            merged[key] = combined
        else:
            if key == "age" or not existing.get(key):
                merged[key] = val

    now = int(time.time())
    with _lock:
        conn = _get_conn()
        conn.execute("""
            INSERT INTO user_profiles
                (device_key, name, age, gender, weight_kg, blood_group,
                 conditions, allergies, medications, location, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(device_key) DO UPDATE SET
                name        = excluded.name,
                age         = excluded.age,
                gender      = excluded.gender,
                weight_kg   = excluded.weight_kg,
                blood_group = excluded.blood_group,
                conditions  = excluded.conditions,
                allergies   = excluded.allergies,
                medications = excluded.medications,
                location    = excluded.location,
                updated_at  = excluded.updated_at
        """, (
            device_key,
            merged.get("name"),
            merged.get("age"),
            merged.get("gender"),
            merged.get("weight_kg"),
            merged.get("blood_group"),
            json.dumps(merged.get("conditions") or []),
            json.dumps(merged.get("allergies") or []),
            json.dumps(merged.get("medications") or []),
            merged.get("location"),
            now,
        ))
        conn.commit()
        conn.close()


def profile_to_context(profile: dict) -> str:
    """
    Format the user profile as a concise context block for injection into the LLM system prompt.
    Returns '' if the profile is empty.
    """
    if not profile:
        return ""
    parts = []
    if profile.get("name"):
        parts.append(f"Name: {profile['name']}")
    if profile.get("age"):
        parts.append(f"Age: {profile['age']} years old")
    if profile.get("gender"):
        parts.append(f"Gender: {profile['gender']}")
    if profile.get("weight_kg"):
        parts.append(f"Weight: {profile['weight_kg']} kg")
    if profile.get("blood_group"):
        parts.append(f"Blood group: {profile['blood_group']}")
    if profile.get("conditions"):
        parts.append(f"Known conditions: {', '.join(profile['conditions'])}")
    if profile.get("allergies"):
        parts.append(f"Known allergies: {', '.join(profile['allergies'])}")
    if profile.get("medications"):
        parts.append(f"Current medications: {', '.join(profile['medications'])}")
    if profile.get("location"):
        parts.append(f"Location: {profile['location']}")
    if not parts:
        return ""
    lines = "\n".join(f"  • {p}" for p in parts)
    return f"[This user's known profile — use to personalize all advice:\n{lines}\nDo NOT ask for info already listed above.]"


# ── Passive extraction ────────────────────────────────────────────────────────

def extract_profile_facts(text: str) -> dict:
    """
    Extract personal facts from a single user message (English text).
    Returns a dict of found facts; only includes keys where something was found.
    """
    if not text or not text.strip():
        return {}
    t = text.lower()
    facts: dict = {}

    # ── Age ──────────────────────────────────────────────────────────────────
    age_patterns = [
        r"i(?:\'m| am)\s+(\d+)\s+years?\s*old",
        r"i(?:\'m| am)\s+(\d+)\s+(?:year|yr)s?",
        r"(\d+)\s+years?\s*old",
        r"my\s+age\s+is\s+(\d+)",
        r"age[:\s]+(\d+)\b",
        r"aged?\s+(\d+)\b",
        r"i(?:\'m| am)\s+(\d+)\b",   # "I am 45" — least specific, checked last
    ]
    for pat in age_patterns:
        m = re.search(pat, t)
        if m:
            age = int(m.group(1))
            if 1 <= age <= 120:
                facts["age"] = age
                break

    # ── Name ────────────────────────────────────────────────────────────────
    name_patterns = [
        r"my\s+name\s+is\s+([a-z][a-z\s]{1,30}?)(?:\s*[,.]|$)",
        r"i(?:\'m| am)\s+called\s+([a-z][a-z\s]{1,20}?)(?:\s*[,.]|$)",
        r"call\s+me\s+([a-z][a-z]{2,20})\b",
        r"name[:\s]+([a-z][a-z\s]{1,25}?)(?:\s*[,.]|$)",
    ]
    _NAME_EXCLUDE = {
        "feeling", "having", "not", "very", "really", "just", "trying", "going",
        "doing", "sorry", "worried", "fine", "okay", "good", "bad", "sick",
        "suffering", "experiencing", "getting", "taking", "asking", "here",
        "a", "an", "the", "ok", "well", "still", "also", "so", "now",
        "diabetic", "hypertensive", "asthmatic", "pregnant", "depressed",
        "anxious", "tired", "happy", "sad", "back", "again", "unsure",
        "confused", "scared", "nervous", "male", "female", "man", "woman",
        "unable", "afraid", "ready", "sure", "concerned", "worried",
    }
    name_patterns.append(r"i(?:\'m| am)\s+([a-z][a-z]{2,18})\s*,")
    for pat in name_patterns:
        m = re.search(pat, t)
        if m:
            candidate = m.group(1).strip().title()
            first_word = candidate.lower().split()[0] if candidate else ""
            if first_word not in _NAME_EXCLUDE and 2 <= len(candidate) <= 40:
                facts["name"] = candidate
                break

    # ── Gender ───────────────────────────────────────────────────────────────
    if re.search(
        r"\bi(?:\'m| am)\s+a?\s*(?:woman|female|girl|mother|wife|sister|aunt|grandmother|widow|pregnant)\b"
        r"|\bas\s+a\s+(?:woman|mother|wife|female|girl)\b"
        r"|\bher\s+(?:age|name|weight)\b",
        t
    ):
        facts["gender"] = "female"
    elif re.search(
        r"\bi(?:\'m| am)\s+a?\s*(?:man|male|boy|father|husband|brother|uncle|grandfather)\b"
        r"|\bas\s+a\s+(?:man|father|husband|male)\b"
        r"|\bhis\s+(?:age|name|weight)\b",
        t
    ):
        facts["gender"] = "male"

    # ── Weight ───────────────────────────────────────────────────────────────
    m = re.search(
        r"(?:i\s+weigh(?:ing)?|my\s+weight(?:\s+is)?|weight[:\s]+|weighs?\s+|\bweigh(?:ing)?\s+)\s*(\d+(?:\.\d+)?)\s*(kg|kilogram|lb|pound)",
        t
    )
    if m:
        w, unit = float(m.group(1)), m.group(2)
        if unit.startswith("lb") or unit.startswith("pound"):
            w = round(w * 0.453592, 1)
        if 3.0 <= w <= 350.0:
            facts["weight_kg"] = w

    # ── Blood group ──────────────────────────────────────────────────────────
    m = re.search(
        r"blood\s*(?:group|type)\s*(?:is)?\s*([AaBbOo]{1,2}\s*[+\-]|ab\s*[+\-])",
        t
    )
    if m:
        facts["blood_group"] = m.group(1).upper().replace(" ", "")

    # ── Known medical conditions ──────────────────────────────────────────────
    _CONDITION_MAP = [
        (r"\b(?:type\s*[12]\s*diabet\w+|diabet(?:ic|es))\b",        "diabetes"),
        (r"\b(?:hypertension|high\s*blood\s*pressure|hypertensive)\b", "hypertension"),
        (r"\b(?:asthma|asthmatic)\b",                                "asthma"),
        (r"\b(?:heart\s*(?:disease|failure|attack|problem)|coronary|angina|cardiac)\b", "heart disease"),
        (r"\b(?:kidney\s*(?:disease|failure|problem|stone)|chronic\s*kidney|ckd)\b", "kidney disease"),
        (r"\b(?:hypothyroid\w*|hyperthyroid\w*|thyroid\s*(?:disease|problem|disorder))\b", "thyroid disease"),
        (r"\b(?:epilep\w+|seizure\s*disorder)\b",                    "epilepsy"),
        (r"\b(?:liver\s*(?:disease|failure|problem|cirrhosis)|hepatitis)\b", "liver disease"),
        (r"\b(?:copd|chronic\s*obstructive\s*pulmonary)\b",          "COPD"),
        (r"\b(?:depression|major\s*depressive)\b",                   "depression"),
        (r"\b(?:anxiety\s*disorder|generalized\s*anxiety|panic\s*disorder)\b", "anxiety disorder"),
        (r"\b(?:i(?:\'m| am)\s+pregnant|pregnancy|gestational)\b",   "pregnant"),
        (r"\b(?:pcos|polycystic\s*ovary)\b",                         "PCOS"),
        (r"\b(?:rheumatoid\s*arthritis)\b",                          "rheumatoid arthritis"),
        (r"\b(?:lupus|sle)\b",                                       "lupus"),
        (r"\b(?:hiv|aids)\b",                                        "HIV"),
        (r"\b(?:cancer|tumor|tumour|malignancy|chemotherapy|oncology)\b", "cancer"),
        (r"\b(?:stroke|tia|mini[\s-]stroke)\b",                      "stroke history"),
        (r"\b(?:migraine\w*)\b",                                     "migraines"),
        (r"\b(?:osteoporosis)\b",                                     "osteoporosis"),
        (r"\b(?:gout)\b",                                            "gout"),
        (r"\b(?:psoriasis)\b",                                       "psoriasis"),
        (r"\b(?:eczema|atopic\s*dermatitis)\b",                      "eczema"),
    ]
    found_conditions = []
    for pat, label in _CONDITION_MAP:
        if re.search(pat, t, re.IGNORECASE):
            found_conditions.append(label)
    if found_conditions:
        facts["conditions"] = found_conditions

    # ── Allergies ────────────────────────────────────────────────────────────
    allergy_hits = []
    for m in re.finditer(
        r"(?:allergic\s+to|allergy\s+to|have\s+(?:an?\s+)?allergy\s+to|react\s+(?:badly\s+)?to)\s+"
        r"([a-z][a-z\s]{1,40}?)(?:\.|,|\band\b|$)",
        t
    ):
        item = m.group(1).strip().rstrip(",. ")
        if item and len(item) < 50:
            allergy_hits.append(item)
    if allergy_hits:
        facts["allergies"] = allergy_hits

    # ── Current medications ──────────────────────────────────────────────────
    med_hits = []
    _MED_STOPWORDS = {
        "care", "medication", "medicine", "tablets", "pills", "some",
        "this", "that", "it", "them", "a", "an", "the", "my", "daily",
        "regularly", "now", "currently", "already"
    }
    for m in re.finditer(
        r"(?:i(?:\'m| am)?\s*(?:currently\s+)?(?:taking|on|using)"
        r"|i\s+have\s+been\s+taking"
        r"|[,;]\s*taking"
        r"|prescribed|take\s+|started\s+(?:on\s+)?)\s*"
        r"([a-z][a-z0-9\s]{1,35}?)(?:\s+(?:for|to|since|daily|tablet|mg|dose|\d)|[,.]|$)",
        t
    ):
        med = re.sub(r"\s*\d[\d.]*\s*(?:mg|mcg|ml|g)?\s*$", "", m.group(1)).strip().rstrip(",. ")
        first = med.split()[0] if med else ""
        if first and first not in _MED_STOPWORDS and len(med) > 2:
            med_hits.append(med)
    if med_hits:
        facts["medications"] = med_hits

    # ── Location (city/district) ─────────────────────────────────────────────
    _BD_LOCATIONS = [
        "dhaka", "chittagong", "chattogram", "sylhet", "rajshahi", "khulna",
        "barisal", "barishal", "mymensingh", "rangpur", "comilla", "cumilla",
        "narayanganj", "gazipur", "tongi", "cox's bazar", "cox bazar",
        "jessore", "jashore", "bogra", "bogura",
    ]
    for loc in _BD_LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", t):
            facts["location"] = loc.title()
            break

    return facts
