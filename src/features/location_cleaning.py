#src/features/location_cleaning.py

import re

# ---------------------------
# Controlled vocabularies
# ---------------------------
JUNK_PREFIXES = [
    "strong", "percent", "asia", "year", "week", "tuesday", "monday"
]

PERSON_TERMS = {
    "khan", "murad", "poonam", "rafael", "radwanska", "sharapova",
    "masakadza", "malinga", "uthappa"
}

ORG_TERMS = {
    "city", "university", "cup", "league", "club"
}

REGIONS = {
    "asia", "south asia", "middle east", "europe",
    "north america", "africa", "latin america"
}


# ---------------------------
# Core cleaner
# ---------------------------
def clean_location(raw):
    if not isinstance(raw, str) or raw.strip() == "":
        return "UNKNOWN", "UNKNOWN"

    loc = raw.lower().strip()

    # Remove numbers & symbols
    loc = re.sub(r"[^a-z\s]", " ", loc)
    loc = re.sub(r"\s+", " ", loc)

    # Remove junk prefixes
    for junk in JUNK_PREFIXES:
        if loc.startswith(junk):
            loc = loc.replace(junk, "").strip()

    # Single-word person names
    if loc in PERSON_TERMS:
        return "UNKNOWN", "UNKNOWN"

    # Org contamination
    if any(term in loc for term in ORG_TERMS):
        return "UNKNOWN", "UNKNOWN"

    # Regions
    if loc in REGIONS:
        return loc.title(), "REGION"

    # Very short tokens
    if len(loc) < 4:
        return "UNKNOWN", "UNKNOWN"

    # Title case final
    return loc.title(), "CITY_OR_COUNTRY"
