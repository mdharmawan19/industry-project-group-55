"""One shared listings table for every website and every year.

Each source (Domain 2025, rent.com.au 2026, ...) has a small "loader" that renames
its columns to the SHARED_COLUMNS below. build_listings() stacks every loader whose
file exists, cleans them the same way and saves data/curated/listings.parquet.
Downstream notebooks only read that file and group by `snapshot`, so adding a new
year means adding one loader here and nothing else.
"""
import re

import numpy as np
import pandas as pd

import config

# Fixed feature list: the same yes/no columns for every source. Each pattern is matched
# against the lower-cased feature text, so it works for Domain wording ("Built in
# wardrobes", "airConditioning") and rent.com.au wording ("Floorboards flooring").
FEATURE_PATTERNS = {
    "air_conditioning": r"air.?con|split.?system|reverse.?cycle|ducted.?cooling|evaporative",
    "heating": r"heating|heater|fireplace",
    "dishwasher": r"dishwasher",
    "built_in_wardrobes": r"built.?in.?wardrobe",
    "balcony_deck": r"balcon|\bdeck",
    "courtyard_garden": r"courtyard|garden|fully.?fenced",
    "secure_parking": r"secure.?parking|remote.?garage|garage",
    "ensuite": r"ensuite",
    "study": r"\bstudy",
    "pool": r"swimming.?pool|pool(?!/spa count: 0)",   # "Pool/Spa Count: 0" means NO pool
    "gym": r"\bgym|fitness",
    "furnished": r"(?<!un)furnished",
    "pets_allowed": r"pets?.?allowed|pet.?friendly",
    "floorboards": r"floorboard|timber.?floor",
    "internal_laundry": r"laundry",
    "solar": r"solar",
    "security": r"intercom|alarm|security",
    "outdoor_entertaining": r"outdoor.?entertain|entertaining|bbq",
}
FEATURE_COLUMNS = [f"has_{name}" for name in FEATURE_PATTERNS]

# Property types from any source -> a few comparable groups
PROPERTY_GROUPS = {
    "House": "House", "Villa": "House", "Semi-Detached": "House", "Semi-detached": "House",
    "Terrace": "House", "Duplex": "House", "Cottage": "House", "New House & Land": "House",
    "Acreage / Semi-Rural": "House", "Acreage": "House", "Rural": "House",
    "Townhouse": "Townhouse",
    "Apartment / Unit / Flat": "Apartment/Unit", "Apartment": "Apartment/Unit",
    "Unit": "Apartment/Unit", "Flat": "Apartment/Unit", "Block of Units": "Apartment/Unit",
    "New Apartments / Off the Plan": "Apartment/Unit", "Penthouse": "Apartment/Unit",
    "Loft": "Apartment/Unit", "Serviced Apartment": "Apartment/Unit",
    "Studio": "Studio",
}

SHARED_COLUMNS = [
    "listing_id", "source", "snapshot", "scraped_date",
    "suburb", "postcode", "suburb_key", "address", "lat", "lon",
    "weekly_rent", "bond", "bedrooms", "bathrooms", "carspaces",
    "property_type", "property_group", "date_listed", "days_listed", "agency",
    "features_text", "walk_score", "transit_score",
]


def suburb_key(suburb, postcode):
    """'Box Hill', 3128 -> 'BOX_HILL_3128' (same key as the DFFH and SA2 crosswalks)."""
    s = suburb.astype("string").str.strip().str.upper().str.replace(r"\s+", "_", regex=True)
    return s + "_" + postcode.astype("string")


def _finish(df, source, snapshot):
    """Shared type conversions for any loader's output."""
    df = df.copy()
    df["source"] = source
    df["snapshot"] = snapshot
    df["listing_id"] = source + "-" + df["listing_id"].astype("string")
    df["postcode"] = (pd.to_numeric(df["postcode"], errors="coerce").astype("Int64")
                      .astype("string").str.zfill(4))
    df["suburb"] = df["suburb"].astype("string").str.strip().str.upper()
    df["suburb_key"] = suburb_key(df["suburb"], df["postcode"])
    for col in ["weekly_rent", "bond", "bedrooms", "bathrooms", "carspaces", "lat", "lon",
                "days_listed", "walk_score", "transit_score"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    for col in ["scraped_date", "date_listed"]:
        df[col] = pd.to_datetime(df.get(col), errors="coerce", utc=True).dt.tz_localize(None)
    df["property_group"] = df["property_type"].map(PROPERTY_GROUPS).fillna("Other")
    return df.reindex(columns=SHARED_COLUMNS)


def load_domain_2025(path=config.DOMAIN_2025):
    d = pd.read_csv(path, low_memory=False)
    d = d.rename(columns={"structured_features": "features_text"})
    return _finish(d, "domain", "2025-09")


def load_rentcomau_2026(path=config.RENTCOMAU_2026):
    """rent.com.au merge output (scripts/scrape_rentcomau.py merge) -> shared columns."""
    r = pd.read_csv(path, low_memory=False, dtype={"listing_id": str})
    if "in_searched_suburb" in r:          # keep listings inside the suburb that was searched
        r = r[r["in_searched_suburb"].astype(str).str.lower() == "true"]
    r = r.rename(columns={"structured_features": "features_text",
                          "activated_at": "date_listed"})
    if "date_listed" in r and "days_listed" not in r:
        scraped = pd.to_datetime(r["scraped_date"], errors="coerce")
        listed = pd.to_datetime(r["date_listed"], errors="coerce", utc=True).dt.tz_localize(None)
        r["days_listed"] = (scraped - listed).dt.days
    if "pets_allowed" in r:                # rent.com.au gives pets as its own column
        pets = r["pets_allowed"].astype(str).str.lower() == "true"
        r["features_text"] = r.get("features_text", pd.Series("", index=r.index)).fillna("")
        r.loc[pets, "features_text"] += ", Pets allowed"
    return _finish(r, "rentcomau", "2026-09")


# Every source the project knows about. A source is skipped if its file isn't there yet.
SOURCES = [
    (config.DOMAIN_2025, load_domain_2025),
    (config.RENTCOMAU_2026, load_rentcomau_2026),
]


def add_features_and_flags(df):
    df = df.copy()
    text = df["features_text"].fillna("").str.lower()
    for name, pattern in FEATURE_PATTERNS.items():
        df[f"has_{name}"] = text.str.contains(pattern, regex=True).astype("int8")
    df["features_missing"] = df["features_text"].isna()

    residential = df["property_group"] != "Other"
    df["flag_rent_suspicious"] = residential & ~df["weekly_rent"].between(100, 10_000)
    df["flag_bad_coordinates"] = ~(df["lat"].between(-39.3, -33.9) & df["lon"].between(140.9, 150.1))
    df["flag_rooms_suspicious"] = (df["bedrooms"] > 20) | ((df["bathrooms"] >= 8) & (df["bedrooms"] <= 2))
    # `usable` = rows fit for rent analysis. Flags are kept so nothing is silently deleted.
    df["usable"] = (residential & df["weekly_rent"].notna() & ~df["flag_rent_suspicious"]
                    & ~df["flag_bad_coordinates"] & ~df["flag_rooms_suspicious"]
                    & df["bedrooms"].between(0, 10))
    return df


def build_listings(save=True):
    frames = [loader(path) for path, loader in SOURCES if path.exists()]
    df = add_features_and_flags(pd.concat(frames, ignore_index=True))
    assert df["listing_id"].is_unique, "duplicate listing ids after stacking sources"
    if save:
        df.to_parquet(config.LISTINGS, index=False)
    return df
