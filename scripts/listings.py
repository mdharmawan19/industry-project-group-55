"""One master table of rental listings for every website and every year.

How it works
  1. Translate: each source has a small loader that renames its columns, converts its
     wording and IDs, and returns the shared columns (BASE_COLUMNS). Source-specific
     handling lives only in the loaders.
  2. Clean: add_features_and_flags() applies ONE set of rules to every source: property
     groups, the 28 yes/no features, the rent rule and the reason flags.
  3. Check and save: validate() refuses to save a table that breaks the column contract;
     build_listings() stacks every source whose file exists and saves
     data/curated/listings.parquet.

notebooks/01_data_curation.ipynb calls build_listings() and explains the rules.
Every later notebook reads the saved table with load_listings() and works per `snapshot`,
so adding a new year means adding one loader to SOURCES and nothing else.

Nothing is deleted: rows that fail a check are flagged, `exclusion_reason` says why,
and analyses use the rows where `usable` is True.
"""
import re

import numpy as np
import pandas as pd

import config

# --------------------------------------------------------------------------------------
# The 28 features. Each listing's feature list is split into labels, each label is
# tidied (normalise_label), and a feature is "yes" if any label matches its pattern.
# Values: 1 = mentioned, 0 = not mentioned, blank = unknown (see add_features_and_flags).
# --------------------------------------------------------------------------------------
FEATURE_PATTERNS = {
    "air_conditioning": r"air.?con|split.?system|reverse.?cycle|ducted.?cooling|evaporative|cooling",
    "heating": r"\bheat(?:ing|er)s?\b|fire.?place|hydronic",
    "dishwasher": r"dishwasher",
    "built_in_wardrobes": r"built.?in.?(?:wardrobe|robe)|walk.?in.?(?:wardrobe|robe)",
    "balcony_deck": r"balcon|\bdeck",
    "courtyard_garden": r"courtyard|garden|fully.?fenced|backyard",
    "secure_parking": r"secure.?parking|garage",
    "ensuite": r"ensuite",
    "study": r"\bstudy",
    "pool": r"swimming.?pool|in.?ground.?pool|above.?ground.?pool|^pool$|pool.?/.?spa.?count: ?[1-9]",
    "gym": r"\bgym|fitness",
    "furnished": r"^(?:fully )?furnished$",                       # not "partially furnished"
    "pets_allowed": r"pets?.?allowed|pet.?friendly",
    "floorboards": r"floorboard|timber.?floor|hardwood|wood.?floor",
    "internal_laundry": r"laundry",
    "solar": r"solar",
    "security": r"intercom|alarm|security.?(?:system|door|camera|access)|cctv|video.?entry",
    "outdoor_entertaining": r"outdoor.?entertain|entertaining|bbq|barbecue|alfresco",
    "gas_appliances": r"^gas$|gas.?(?:appliance|cook|stove|oven|hot.?water|hotplate|cooktop)",  # not gas heating
    "bath": r"\bbath(?:tub)?s?\b",                                 # not "bathroom"
    "broadband": r"broadband|\bnbn\b|internet|adsl|fibre|wi.?fi",
    "shed": r"\bshed\b",
    "carpet": r"carpet",
    "openable_windows": r"openable.?window",
    "rumpus_room": r"rumpus|games.?room|family.?room",
    "views": r"\bviews?\b",
    "separate_dining": r"separate.?dining|formal.?dining",
    "double_glazing": r"double.?glaz",
}
FEATURE_COLUMNS = [f"has_{name}" for name in FEATURE_PATTERNS]

# Features a source's listing form barely offers. For these, that source's rows are
# "yes if mentioned, otherwise unknown", never "no" (rent.com.au: under 1% vs up to 59% on Domain).
NOT_TAGGED_BY = {
    "rentcomau": ["built_in_wardrobes", "internal_laundry", "solar", "gas_appliances", "bath", "carpet",
                  "openable_windows", "views", "separate_dining", "double_glazing"],
}
# rent.com.au shows these as their own fields, so they are known even without a feature list
FIELD_FEATURES = {"pets_allowed": "_pets", "furnished": "_furnished", "ensuite": "_ensuite"}
# rent.com.au feature groups whose name carries meaning ("Heating: gas" = gas heating)
KEEP_GROUP = {"heating", "airconditioning"}

# --------------------------------------------------------------------------------------
# Property types -> groups. Every residential type is kept (houses AND apartments).
# --------------------------------------------------------------------------------------
PROPERTY_GROUPS = {   # lower-case type as shown by either site -> group
    "house": "House", "villa": "House", "semi-detached": "House", "terrace": "House",
    "duplex": "House", "cottage": "House", "new house & land": "House",
    "acreage / semi-rural": "House", "acreage": "House", "rural": "House", "farm": "House",
    "duplex semi": "House", "house and land": "House",
    "townhouse": "Townhouse",
    "apartment / unit / flat": "Apartment/Unit", "apartment": "Apartment/Unit", "unit": "Apartment/Unit",
    "flat": "Apartment/Unit", "block of units": "Apartment/Unit",
    "new apartments / off the plan": "Apartment/Unit", "penthouse": "Apartment/Unit",
    "loft": "Apartment/Unit", "serviced apartment": "Apartment/Unit", "granny flat": "Apartment/Unit",
    "retirement": "Apartment/Unit", "retirement living": "Apartment/Unit", "warehouse": "Apartment/Unit",
    "studio": "Studio",
}
AMBIGUOUS_TYPES = {"other", "unknown"}          # kept as "Other residential" if 1+ bedroom
NOT_RESIDENTIAL_TYPES = {"car space", "carspace", "car park", "parking", "vacant land", "land",
                         "commercial", "room"}                    # a room's rent is per room
RESIDENTIAL_GROUPS = ["House", "Townhouse", "Apartment/Unit", "Studio", "Other residential"]

# --------------------------------------------------------------------------------------
# Rule A (the one rent rule for the whole project)
# --------------------------------------------------------------------------------------
RENT_RANGE = (100, 10_000)                       # $ per week
MAX_BEDROOMS = 10
VIC_BOX = {"lat": (-39.3, -33.9), "lon": (140.9, 150.1)}

# --------------------------------------------------------------------------------------
# The column contract of listings.parquet
# --------------------------------------------------------------------------------------
BASE_COLUMNS = [
    "listing_id", "source", "snapshot", "scraped_date",
    "suburb", "postcode", "suburb_key", "address", "lat", "lon",
    "weekly_rent", "bond", "bedrooms", "bathrooms", "carspaces",
    "property_type", "property_group", "date_listed", "days_listed", "agency", "photo_count",
    "features_text",
]
TEMP_COLUMNS = ["_features_listed", "_pets", "_furnished", "_ensuite",
                "_available_date"]                                       # loader -> cleaner only
# Rule A checks, in the order they are reported: the first one a listing fails becomes its
# exclusion_reason (the checks themselves are in rule_a_checks()).
REASONS = ["not residential", "rent missing", "rent outside $100-$10,000", "bad coordinates",
           "bedrooms missing", "implausible room counts"]
# The nine review flags from 01_data_curation Part 1 (the teammate's 2025 curation), computed the
# same way for every year. They mark listings for review; Rule A decides `usable`.
FLAG_COLUMNS = ["flag_future_listed", "flag_long_listing", "flag_invalid_available_date",
                "flag_rent_suspicious", "flag_bedrooms_suspicious", "flag_bathrooms_suspicious",
                "flag_carspaces_suspicious", "flag_property_features_suspicious", "flag_bad_coordinates"]
CONTRACT_COLUMNS = (BASE_COLUMNS + FEATURE_COLUMNS + ["features_missing"] + FLAG_COLUMNS
                    + ["exclusion_reason", "usable"])

NUMERIC = ["weekly_rent", "bond", "bedrooms", "bathrooms", "carspaces", "lat", "lon",
           "days_listed", "photo_count"]
TEXT = ["address", "agency", "property_type", "features_text"]


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def suburb_key(suburb, postcode):
    """'Box Hill', '3128' -> 'BOX_HILL_3128' (the key used by the SA2 and DFFH crosswalks)."""
    s = suburb.astype("string").str.strip().str.upper().str.replace(r"\s+", "_", regex=True)
    return s + "_" + postcode.astype("string")


def _col(df, name):
    """A column, or a blank column if this source doesn't have it."""
    return df[name] if name in df else pd.Series(np.nan, index=df.index, dtype="object")


def _yes_no(series):
    """True/False/'True'/'false'/1/0 -> 1.0/0.0, anything else -> blank."""
    text = series.astype("string").str.strip().str.lower()
    return text.map({"true": 1.0, "1": 1.0, "1.0": 1.0, "false": 0.0, "0": 0.0, "0.0": 0.0}).astype(float)


def normalise_label(label):
    """'Built in wardrobes' / 'airConditioning' / 'Balcony-Deck' -> tidy lower-case words."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(label).strip())
    s = s.lower().replace("&", "and")
    s = re.sub(r"[-_]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def rentcom_label(label):
    """rent.com.au shows features as 'Group: item'. Keep the item; keep the group only where
    it carries meaning: 'Heating: gas' -> 'gas heating', 'Security: fully fenced' -> 'fully fenced'."""
    m = re.match(r"^([^:]+):\s*(.+)$", label.strip())
    if not m:
        return label.strip()
    group, item = m.group(1).strip().lower(), m.group(2).strip()
    return f"{item} {group}" if group in KEEP_GROUP else item


def split_labels(features_text):
    """Long table: one tidied label per row, indexed by the listing's row."""
    labels = features_text.fillna("").astype(str).str.split(",").explode().str.strip()
    return labels[labels != ""].map(normalise_label)


# --------------------------------------------------------------------------------------
# Loaders: one per source (the only place source-specific handling happens)
# --------------------------------------------------------------------------------------
def _finish(df, source, snapshot):
    """Shared conversions for any loader's output -> BASE_COLUMNS (+ temp columns)."""
    df = df.copy()
    df["source"] = source
    df["snapshot"] = snapshot
    df["listing_id"] = source + "-" + _col(df, "listing_id").astype("string")
    df["postcode"] = (pd.to_numeric(_col(df, "postcode"), errors="coerce").astype("Int64")
                      .astype("string").str.zfill(4))
    df["suburb"] = _col(df, "suburb").astype("string").str.strip().str.upper()
    df["suburb_key"] = suburb_key(df["suburb"], df["postcode"])
    for c in NUMERIC:
        df[c] = pd.to_numeric(_col(df, c), errors="coerce")
    for c in ["scraped_date", "date_listed"]:
        df[c] = pd.to_datetime(_col(df, c), errors="coerce", utc=True).dt.tz_localize(None)
    for c in TEXT:
        df[c] = _col(df, c).astype("string")
    for c in TEMP_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df.reindex(columns=[c for c in BASE_COLUMNS if c != "property_group"] + TEMP_COLUMNS)


def load_domain_2025(path=config.DOMAIN_2025):
    """Course dataset (Domain.com.au, Sept 2025)."""
    d = pd.read_csv(path, low_memory=False)
    text = _col(d, "structured_features").astype("string")
    d["features_text"] = text
    d["_features_listed"] = text.fillna("").str.strip().ne("")
    d["_available_date"] = _col(d, "available_date")
    return _finish(d, "domain", "2025-09")


def load_rentcomau_2026(path=config.RENTCOMAU_2026):
    """rent.com.au scrape (Sept 2026), the output of `scrape_rentcomau.py merge`.

    The merge already keeps one copy per listing (its own suburb's search when it exists),
    so every row is kept, including listings from suburbs nobody searched."""
    r = pd.read_csv(path, low_memory=False, dtype={"listing_id": str})
    raw = _col(r, "structured_features")
    r["features_text"] = raw.map(
        lambda t: ", ".join(rentcom_label(x) for x in t.split(",") if x.strip())
        if isinstance(t, str) and t.strip() else np.nan)
    # A feature list exists only if the listing had "Group: item" features; the scraper's own
    # extras ("Furnished", "Ensuite") come from separate fields and are handled below.
    r["_features_listed"] = raw.map(lambda t: isinstance(t, str) and ":" in t)
    has_details = _col(r, "detail_scraped_date").notna()
    r["_available_date"] = _col(r, "available_date")
    r["_pets"] = _yes_no(_col(r, "pets_allowed"))
    r["_furnished"] = _yes_no(_col(r, "furnished")).where(has_details)
    ensuites = pd.to_numeric(_col(r, "ensuites"), errors="coerce")
    r["_ensuite"] = (ensuites > 0).astype(float).where(has_details & ensuites.notna())
    if "days_listed" not in r:
        scraped = pd.to_datetime(_col(r, "scraped_date"), errors="coerce")
        listed = pd.to_datetime(_col(r, "date_listed"), errors="coerce")
        r["days_listed"] = (scraped.dt.normalize() - listed).dt.days
    return _finish(r, "rentcomau", "2026-09")


# Every source the project knows about. A source is skipped if its file isn't there yet.
SOURCES = [
    (config.DOMAIN_2025, load_domain_2025),
    (config.RENTCOMAU_2026, load_rentcomau_2026),
]
SNAPSHOTS = {"domain": "2025-09", "rentcomau": "2026-09"}


# --------------------------------------------------------------------------------------
# Cleaning: the same rules for every source
# --------------------------------------------------------------------------------------
def type_key(text):
    """Compare type names ignoring case and punctuation: 'Acreage / Semi-Rural' and
    'Acreage Semi Rural' both become 'acreage semi rural'."""
    s = text.astype("string").str.lower().str.replace(r"[/\-&]", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


_GROUP_LOOKUP = dict(zip(type_key(pd.Series(list(PROPERTY_GROUPS))), PROPERTY_GROUPS.values()))
_NOT_RESIDENTIAL = set(type_key(pd.Series(sorted(NOT_RESIDENTIAL_TYPES))))
_AMBIGUOUS = set(type_key(pd.Series(sorted(AMBIGUOUS_TYPES))))


def property_group(property_type, bedrooms):
    t = type_key(property_type)
    group = t.map(_GROUP_LOOKUP)
    ambiguous = t.isna() | t.isin(_AMBIGUOUS)
    group = group.mask(ambiguous & (bedrooms >= 1), "Other residential")
    return group.fillna("Not residential").astype("string")


def has_valid_coordinates(df):
    return df["lat"].between(*VIC_BOX["lat"]) & df["lon"].between(*VIC_BOX["lon"])


def rule_a_checks(df):
    """Each Rule A check as a True/False column (True = the listing fails it)."""
    rent, beds, baths = df["weekly_rent"], df["bedrooms"], df["bathrooms"]
    checks = {
        "not residential": ~df["property_group"].isin(RESIDENTIAL_GROUPS),
        "rent missing": rent.isna(),
        "rent outside $100-$10,000": rent.notna() & ~rent.between(*RENT_RANGE),
        "bad coordinates": ~has_valid_coordinates(df),
        "bedrooms missing": beds.isna(),
        "implausible room counts": ((beds > MAX_BEDROOMS) | ((baths >= 8) & (beds <= 2))).fillna(False),
    }
    assert list(checks) == REASONS
    return {k: v.astype(bool) for k, v in checks.items()}


def review_flags(df):
    """The nine review flags of 01_data_curation Part 1, for every source. Domain's
    `secondary_type` is replaced by `property_group` (rent.com.au has no secondary type)."""
    rent, beds, baths, cars = df["weekly_rent"], df["bedrooms"], df["bathrooms"], df["carspaces"]
    available = pd.to_datetime(df["_available_date"].astype("string").str.replace(r"^\w+day,\s*", "", regex=True),
                               errors="coerce", format="mixed", dayfirst=True)
    standard = df["property_group"].isin(["House", "Townhouse", "Apartment/Unit", "Studio"])
    flags = {
        "flag_future_listed": df["date_listed"] > df["scraped_date"],
        "flag_long_listing": df["days_listed"] > 365,
        "flag_invalid_available_date": available.dt.year < 2020,
        "flag_rent_suspicious": standard & ((rent < 100) | (rent > 10_000)),
        "flag_bedrooms_suspicious": beds > 20,
        "flag_bathrooms_suspicious": (baths > 10) | ((baths >= 8) & (beds <= 2)),
        "flag_carspaces_suspicious": (cars >= 10) & df["property_group"].isin(["Apartment/Unit", "Studio"]),
        "flag_bad_coordinates": (df["lat"].isna() | df["lon"].isna()
                                 | ~df["lat"].between(-40, -33) | ~df["lon"].between(140, 150)),
    }
    flags["flag_property_features_suspicious"] = (flags["flag_bedrooms_suspicious"]
                                                  | flags["flag_bathrooms_suspicious"]
                                                  | flags["flag_carspaces_suspicious"])
    return {c: flags[c].fillna(False).astype(bool) for c in FLAG_COLUMNS}


def add_features_and_flags(df):
    df = df.copy().reset_index(drop=True)
    df["property_group"] = property_group(df["property_type"], df["bedrooms"])

    # 28 features: 1 if any label matches, else 0
    labels = split_labels(df["features_text"])
    for name, pattern in FEATURE_PATTERNS.items():
        hit = labels.str.contains(pattern, regex=True)
        df[f"has_{name}"] = hit.groupby(level=0).any().reindex(df.index, fill_value=False).astype(float)

    # No feature list at all -> every feature unknown (not "has none of them")
    listed = df["_features_listed"].fillna(False).astype(bool)
    df.loc[~listed, FEATURE_COLUMNS] = np.nan
    df["features_missing"] = ~listed

    # Features a source barely offers: yes if mentioned, otherwise unknown
    for source, names in NOT_TAGGED_BY.items():
        rows = df["source"] == source
        for name in names:
            col = f"has_{name}"
            df.loc[rows & (df[col] != 1), col] = np.nan

    # Features a source shows as their own field are known even without a feature list
    for name, field in FIELD_FEATURES.items():
        known = df[field].notna()
        df.loc[known, f"has_{name}"] = df.loc[known, field].astype(float)

    # Rule A: one flag per check; the first failed check becomes exclusion_reason;
    # usable = passed every check
    checks = rule_a_checks(df)
    for flag, values in review_flags(df).items():
        df[flag] = values
    df["exclusion_reason"] = pd.Series(
        np.select(list(checks.values()), list(checks.keys()), default=""),
        index=df.index).replace("", np.nan).astype("string")
    df["usable"] = df["exclusion_reason"].isna()

    df = df.drop(columns=TEMP_COLUMNS)
    return df.reindex(columns=CONTRACT_COLUMNS)


# --------------------------------------------------------------------------------------
# Checks, building, reading
# --------------------------------------------------------------------------------------
def validate(df):
    """Refuse a table that breaks the contract; return it unchanged if all is well."""
    problems = []
    missing = [c for c in CONTRACT_COLUMNS if c not in df]
    if missing:
        problems.append(f"missing columns: {missing}")
    if not df["listing_id"].is_unique:
        problems.append(f"{df['listing_id'].duplicated().sum()} duplicate listing ids")
    unknown = set(df["snapshot"].dropna()) - set(SNAPSHOTS.values())
    if unknown:
        problems.append(f"unknown snapshots: {sorted(unknown)}")
    keys = df["suburb_key"].dropna()
    bad_keys = keys[~keys.str.fullmatch(r"[A-Z0-9_'\-]+_\d{4}")]
    if len(bad_keys):
        problems.append(f"{len(bad_keys)} badly formed suburb keys, e.g. {bad_keys.iloc[0]!r}")
    for c in FEATURE_COLUMNS:
        if not df[c].dropna().isin([0, 1]).all():
            problems.append(f"{c} has values other than 0, 1 or blank")
    use = df[df["usable"]]
    if use[["weekly_rent", "lat", "lon", "bedrooms"]].isna().any().any():
        problems.append("some usable rows lack rent, coordinates or bedrooms")
    if problems:
        raise ValueError("listings table failed its checks:\n  - " + "\n  - ".join(problems))
    return df


def unknown_types(df):
    """Property types that are neither in PROPERTY_GROUPS nor known non-residential types:
    a new spelling from a website should be added to PROPERTY_GROUPS, not silently dropped."""
    keys = type_key(df["property_type"])
    unknown = keys.notna() & ~keys.isin(_GROUP_LOOKUP) & ~keys.isin(_NOT_RESIDENTIAL) & ~keys.isin(_AMBIGUOUS)
    return df.loc[unknown, "property_type"].value_counts()


def build_listings(save=True, verbose=True):
    """Load every available source, clean with one set of rules, check, and save."""
    frames = [loader(path) for path, loader in SOURCES if path.exists()]
    df = validate(add_features_and_flags(pd.concat(frames, ignore_index=True)))
    new_types = unknown_types(df)
    if len(new_types):
        print("WARNING: property types not in PROPERTY_GROUPS (treated as not residential):",
              new_types.to_dict())
    if save:
        df.to_parquet(config.LISTINGS, index=False)
    if verbose:
        summary = df.groupby(["snapshot", "source"]).agg(
            listings=("listing_id", "size"), usable=("usable", "sum"),
            with_feature_list=("features_missing", lambda s: (~s).sum()))
        print(summary.to_string())
        if save:
            print(f"Saved {len(df):,} listings to {config.LISTINGS.relative_to(config.ROOT)}")
    return df


def load_listings(rebuild=False):
    """The saved master table (built once by 01_data_curation). Rebuilt only if missing."""
    if rebuild or not config.LISTINGS.exists():
        return build_listings()
    return pd.read_parquet(config.LISTINGS)


def feature_report(df):
    """Every tidied feature label per source, how many listings use it, and which of the 28
    it counts towards ('unmatched' if none). Used in 01 to show nothing slips through."""
    labels = split_labels(df["features_text"])
    long = pd.DataFrame({"source": df["source"].reindex(labels.index).values, "label": labels.values,
                         "row": labels.index})
    matches = {lab: ", ".join(n for n, p in FEATURE_PATTERNS.items() if re.search(p, lab)) or "unmatched"
               for lab in long["label"].unique()}
    long["counts_towards"] = long["label"].map(matches)
    return (long.groupby(["source", "label", "counts_towards"])["row"].nunique()
                .rename("listings").reset_index()
                .sort_values(["source", "listings"], ascending=[True, False], ignore_index=True))


def type_report(df):
    """Every property type as shown by each site and the group it went to."""
    return (df.groupby(["source", "property_type", "property_group"], dropna=False)
              .size().rename("listings").reset_index()
              .sort_values(["source", "listings"], ascending=[True, False], ignore_index=True))
