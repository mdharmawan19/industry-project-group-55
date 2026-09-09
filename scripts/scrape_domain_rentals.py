"""
Sprint 1 Domain rental scraper for MAST30034 Project 2.

Purpose
-------
Collect a small, reproducible sample of publicly visible Victorian rental
listings from Domain.com.au for the Week 6 scraping proof of concept.

The script:
1. requests one search-results page per chosen suburb;
2. extracts the listing URL/ID, rent, address, suburb/postcode,
   bedrooms, bathrooms, car spaces, and property type;
3. keeps both raw text and parsed numeric values where useful;
4. saves the result to CSV.

It deliberately:
- does not log in;
- does not bypass access controls;
- does not use proxies/stealth tooling;
- does not scrape individual listing pages;
- pauses between requests.

Run from your project repository, for example:

    python scripts/scrape_domain_rentals.py

Or choose a smaller test:

    python scripts/scrape_domain_rentals.py --locations melbourne-vic-3000

Or several suburbs:

    python scripts/scrape_domain_rentals.py \
        --locations melbourne-vic-3000 south-yarra-vic-3141 footscray-vic-3011

Dependencies:
    pip install requests beautifulsoup4 pandas
"""

from __future__ import annotations

import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup, Tag


BASE_URL = "https://www.domain.com.au"

# Small Sprint 1 development sample. One results page is requested per suburb.
DEFAULT_LOCATIONS = [
    "melbourne-vic-3000",
    "south-yarra-vic-3141",
    "footscray-vic-3011",
    "clayton-vic-3168",
    "geelong-vic-3220",
]

DEFAULT_OUTPUT = Path("data/raw/domain_rentals_sprint1.csv")

HEADERS = {
    # A normal descriptive user-agent; no browser impersonation/stealth measures.
    "User-Agent": (
        "Mozilla/5.0 (compatible; MAST30034-student-project/1.0; "
        "+educational-use)"
    ),
    "Accept-Language": "en-AU,en;q=0.9",
}

# Current Domain listing URLs end with a numeric listing id, e.g.
# .../311-29-market-street-melbourne-vic-3000-18256960
LISTING_URL_RE = re.compile(r"/[^?#]+-(\d{7,12})(?:[/?#]|$)")

RENT_RE = re.compile(
    r"\$\s*([\d,]+(?:\.\d+)?)\s*"
    r"(?:per\s*week|pw|p/w|weekly)",
    flags=re.IGNORECASE,
)

BED_RE = re.compile(r"(\d+|[-−])\s*Beds?\b", flags=re.IGNORECASE)
BATH_RE = re.compile(r"(\d+|[-−])\s*Baths?\b", flags=re.IGNORECASE)
PARK_RE = re.compile(r"(\d+|[-−])\s*Parking\b", flags=re.IGNORECASE)

PROPERTY_TYPES = [
    "Apartment / Unit / Flat",
    "House",
    "Townhouse",
    "Studio",
    "Villa",
    "Semi-Detached",
    "Terrace",
    "Duplex",
    "Acreage / Semi-Rural",
    "Retirement Living",
]


def clean_space(value: str) -> str:
    """Collapse repeated whitespace into one space."""
    return re.sub(r"\s+", " ", value).strip()


def parse_count(match: re.Match[str] | None) -> int | None:
    """Convert a feature count to int; Domain's dash means not stated."""
    if match is None:
        return None

    value = match.group(1)
    if value in {"-", "−"}:
        return None

    return int(value)


def find_property_type(card_text: str) -> str | None:
    """Return the first recognised property type present in a listing card."""
    lower = card_text.lower()

    # Longest first prevents a generic term from winning accidentally.
    for property_type in sorted(PROPERTY_TYPES, key=len, reverse=True):
        if property_type.lower() in lower:
            return property_type

    return None


def find_listing_card(anchor: Tag) -> Tag | None:
    """
    Walk upward from an address link until the smallest parent containing
    both rental-price text and bedroom text is found.

    This avoids depending on generated CSS class names.
    """
    parent = anchor

    for _ in range(8):
        parent = parent.parent

        if not isinstance(parent, Tag):
            return None

        text = clean_space(parent.get_text(" ", strip=True))

        if RENT_RE.search(text) and BED_RE.search(text):
            return parent

    return None


def parse_address_suburb(address_text: str) -> tuple[str, str | None]:
    """
    Domain result links currently show text such as:
        '311/29 Market Street, Melbourne'

    Keep the full text as address and use the final comma-separated
    component as the displayed suburb.
    """
    address_text = clean_space(address_text)

    if "," not in address_text:
        return address_text, None

    parts = [part.strip() for part in address_text.split(",")]
    return address_text, parts[-1] or None


def postcode_from_url(url: str) -> str | None:
    """Extract VIC postcode from the canonical listing URL when present."""
    match = re.search(r"-vic-(\d{4})-\d{7,12}(?:[/?#]|$)", url, re.IGNORECASE)
    return match.group(1) if match else None


def listing_id_from_url(url: str) -> str | None:
    """Extract Domain listing ID from URL."""
    match = LISTING_URL_RE.search(url)
    return match.group(1) if match else None


def scrape_search_page(
    session: requests.Session,
    location_slug: str,
    timeout: int = 30,
) -> list[dict]:
    """Scrape one Domain rental search-results page for one location."""
    search_url = f"{BASE_URL}/rent/{location_slug}/"

    response = session.get(search_url, headers=HEADERS, timeout=timeout)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    records: list[dict] = []
    seen_urls: set[str] = set()

    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "")

        if not LISTING_URL_RE.search(href):
            continue

        listing_url = urljoin(BASE_URL, href)

        if listing_url in seen_urls:
            continue

        card = find_listing_card(anchor)
        if card is None:
            continue

        card_text = clean_space(card.get_text(" ", strip=True))
        address_text = clean_space(anchor.get_text(" ", strip=True))

        # A property URL may be linked by images as well as by the address.
        # Only use anchors that contain meaningful address-like text.
        if not address_text or len(address_text) < 5:
            continue

        rent_match = RENT_RE.search(card_text)
        bed_match = BED_RE.search(card_text)
        bath_match = BATH_RE.search(card_text)
        park_match = PARK_RE.search(card_text)

        # Require the key target + bedroom field for this Sprint 1 sample.
        if rent_match is None or bed_match is None:
            continue

        address, displayed_suburb = parse_address_suburb(address_text)

        weekly_rent_raw = rent_match.group(0)
        weekly_rent = float(rent_match.group(1).replace(",", ""))

        # Most rents are integer AUD amounts; store integers where possible.
        if weekly_rent.is_integer():
            weekly_rent = int(weekly_rent)

        records.append(
            {
                "listing_id": listing_id_from_url(listing_url),
                "url": listing_url,
                "weekly_rent_raw": weekly_rent_raw,
                "weekly_rent": weekly_rent,
                "address": address,
                "suburb": displayed_suburb,
                "postcode": postcode_from_url(listing_url),
                "features_raw": " ".join(
                    match.group(0)
                    for match in (bed_match, bath_match, park_match)
                    if match is not None
                ),
                "bedrooms": parse_count(bed_match),
                "bathrooms": parse_count(bath_match),
                "carspaces": parse_count(park_match),
                "property_type": find_property_type(card_text),
                "search_location": location_slug,
                "source_page": search_url,
                "scraped_date": datetime.now(timezone.utc).isoformat(),
            }
        )

        seen_urls.add(listing_url)

    return records


def validate_output(df: pd.DataFrame) -> None:
    """
    Fail loudly when extraction clearly did not work.
    This is preferable to silently saving an empty/broken dataset.
    """
    if df.empty:
        raise RuntimeError(
            "No listings were extracted. Domain may have changed its page "
            "structure, blocked the request, or returned a challenge page."
        )

    required = ["listing_id", "weekly_rent", "bedrooms", "url"]

    for column in required:
        if column not in df.columns:
            raise RuntimeError(f"Expected column missing: {column}")

    if df["listing_id"].notna().sum() == 0:
        raise RuntimeError("No listing IDs were parsed.")

    if df["weekly_rent"].notna().sum() == 0:
        raise RuntimeError("No weekly rents were parsed.")

    if df["bedrooms"].notna().sum() == 0:
        raise RuntimeError("No bedroom counts were parsed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape a small Sprint 1 sample of Domain VIC rentals."
    )
    parser.add_argument(
        "--locations",
        nargs="+",
        default=DEFAULT_LOCATIONS,
        help=(
            "Domain location slugs, e.g. melbourne-vic-3000 "
            "south-yarra-vic-3141"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to pause between suburb requests (default: 2).",
    )
    args = parser.parse_args()

    session = requests.Session()

    all_records: list[dict] = []

    for i, location in enumerate(args.locations, start=1):
        print(f"[{i}/{len(args.locations)}] {location}")

        try:
            records = scrape_search_page(session, location)
            print(f"  extracted {len(records)} listings")
            all_records.extend(records)

        except requests.RequestException as exc:
            print(f"  request failed: {exc}")

        if i < len(args.locations):
            time.sleep(args.delay)

    df = pd.DataFrame(all_records)

    if not df.empty:
        # The same listing can occasionally appear in more than one search.
        df = (
            df.drop_duplicates(subset=["listing_id"], keep="first")
            .reset_index(drop=True)
        )

    validate_output(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print("\nSaved:", args.output)
    print("Rows:", len(df))
    print("Suburbs represented:", df["suburb"].nunique(dropna=True))
    print("\nMissing values in key fields:")
    print(
        df[
            [
                "weekly_rent",
                "bedrooms",
                "bathrooms",
                "carspaces",
                "property_type",
                "postcode",
            ]
        ]
        .isna()
        .sum()
        .to_string()
    )

    print("\nFirst five rows:")
    print(
        df[
            [
                "suburb",
                "postcode",
                "weekly_rent",
                "bedrooms",
                "bathrooms",
                "carspaces",
                "property_type",
            ]
        ]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
