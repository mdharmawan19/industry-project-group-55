#!/usr/bin/env python3
"""
Polite, shareable scraper for Victorian rental listings on rent.com.au (MAST30034 Project 2).

The suburb list is split into shares so several people can scrape at the same time, each on
their own computer. Everyone's files are then combined with `merge`.

Stages
  plan      Split the suburbs between N people and print each person's command.
  test      2 requests (1 search page + 1 listing page) to check everything works.
            Add --offline to check the parser on the saved pages without using the internet.
  search    Suburb search pages (25 listings per page). Rent, rooms, type, address,
            coordinates, listing date, available date and agency all come from here.
  details   OPTIONAL. One request per listing for bond, features, furnished and description.
            Slow (about 5 s per listing) but resumable.
  status    Progress for each person's share.
  merge     Combine everyone's files into one CSV with the Domain dataset's column names.
            This is the file the notebooks read (see scripts/listings.py).

Typical use (from the repo's top folder):
  python3 scripts/scrape_rentcomau.py plan --workers 4
  python3 scripts/scrape_rentcomau.py test --contact you@student.unimelb.edu.au
  python3 scripts/scrape_rentcomau.py search --shard 2/4 --worker kerri --contact you@student.unimelb.edu.au
  python3 scripts/scrape_rentcomau.py status --workers 4
  python3 scripts/scrape_rentcomau.py merge

Being a good guest on the site:
  - Reads robots.txt first and never requests a page it disallows.
  - Waits about 5 seconds between requests (never less than 3), longer if robots.txt asks.
  - Says who it is in the User-Agent (a student project, with your contact email).
  - Stops straight away if the site refuses access (403 / 429 / 503 / "checking your browser").
    It does not try to get around blocking: if that happens, stop and use the course dataset.
  - Only whitelisted listing fields are saved. Agents' and landlords' names and phone numbers,
    which the pages also contain, are never written to disk.

Output (ignored by git): data/raw/rentcomau/<snapshot>/
  search/<worker>/<Suburb>_<postcode>.csv   one file per suburb (a suburb with a file is done)
  details/<worker>.csv                      optional listing-page details
  vic_rentals_rentcomau.csv                 merged result

Data source: rent.com.au, collected for MAST30034 coursework only. Do not share the raw data
outside the group or commit it to GitHub.
"""

import argparse
import csv
import json
import math
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402  (repo paths; works from any folder)

BASE = "https://www.rent.com.au"
PER_PAGE = 25
DEFAULT_MAX_PAGES = 12              # 12 x 25 = 300 per suburb, the same cap as the course dataset
DEFAULT_SNAPSHOT = "2026-09"        # everyone in the group uses the same snapshot name
UA_TOKEN = "MAST30034-student-project"
SUBURBS = config.RAW / "domain" / "Data" / "suburb_summary.csv"
RENT_DIR = config.RAW / "rentcomau"
DEBUG_DIR = RENT_DIR / "debug"
PROPERTY_HREF = re.compile(r"/property/[^/?#]*?-(\d+)/?$")
WORKER_NAME = re.compile(r"^[a-z0-9_-]{1,30}$")

SEARCH_COLUMNS = [
    "listing_id", "url", "address", "suburb", "postcode", "weekly_rent", "price_text",
    "bedrooms", "bathrooms", "carspaces", "property_type", "lat", "lon", "date_listed",
    "available_date", "pets_allowed", "agency", "walk_score", "transit_score",
    "search_suburb", "search_postcode", "in_searched_suburb", "page", "parse_method",
    "worker", "scraped_date",
]
DETAIL_COLUMNS = [
    "listing_id", "bond", "weekly_rent_detail", "furnished", "partially_furnished", "ensuites",
    "structured_features", "byline", "description", "photo_count", "detail_scraped_date",
]
# Merged file: Domain dataset column names first, rent.com.au extras after
MERGED_COLUMNS = [
    "listing_id", "suburb", "postcode", "weekly_rent", "bond", "available_date", "date_listed",
    "days_listed", "bedrooms", "bathrooms", "carspaces", "property_type", "address", "lat", "lon",
    "scraped_date", "photo_count", "agency", "structured_features", "url",
    "price_text", "pets_allowed", "furnished", "ensuites", "walk_score", "transit_score",
    "byline", "description", "in_searched_suburb", "search_suburb", "search_postcode",
    "worker", "parse_method", "detail_scraped_date", "snapshot", "source",
]


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
class Blocked(Exception):
    pass


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def squash(text):
    return " ".join(str(text).split()) if text is not None else ""


def to_int(pattern, text):
    m = re.search(pattern, text or "", re.I)
    return int(m.group(1)) if m else None


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def money(s):
    return float(s.replace(",", ""))


def parse_weekly_rent(text):
    """Weekly rent in AUD from free text, or None.

    "$680 pw" -> 680, "$550 - $600 pw" -> 575, "$330/week, $1434pcm" -> 330,
    "$2,400 per month" -> 553.85, "$31,200 p.a." -> 600, "Contact agent" -> None.
    Values outside $50-$20,000 a week are treated as errors (None).
    """
    if not text:
        return None
    t = str(text).lower()
    num = r"\$\s*([\d,]+(?:\.\d+)?)"
    week = r"\s*(?:pw\b|p/w|/\s?w(?:ee)?k?\b|per\s*week|weekly|a\s*week)"
    month = r"\s*(?:pcm\b|p\.?c\.?m\.?|/\s?m(?:on)?th\b|/\s?month|per\s*(?:calendar\s*)?month|monthly)"
    year = r"\s*(?:p\.?a\.?(?=\W|$)|per\s*(?:annum|year)|/\s?(?:year|yr)|yearly|annual)"

    val = None
    rng = re.search(num + r"\s*(?:-|–|to)\s*" + num, t)
    if rng:
        val = (money(rng.group(1)) + money(rng.group(2))) / 2
        if re.search(month, t[rng.end():rng.end() + 25]):
            val = val * 12 / 52
    elif m := re.search(num + week, t):
        val = money(m.group(1))
    elif m := re.search(num + month, t):
        val = money(m.group(1)) * 12 / 52
    elif m := re.search(num + year, t):
        val = money(m.group(1)) / 52
    elif m := re.search(num, t):
        val = money(m.group(1))       # no unit given: rent.com.au shows weekly by default

    if val is None or not (50 <= val <= 20000):
        return None
    return round(val, 2)


def parse_date(text):
    """'5th October 2026' -> '2026-10-05'; 'now' stays 'now'; anything else is kept as text."""
    if not text:
        return None
    text = squash(text)
    if text.lower() in ("now", "available now"):
        return "now"
    cleaned = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    try:
        return pd.to_datetime(cleaned, dayfirst=True).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return text


def suburb_slug(suburb, postcode):
    return f"{re.sub(r'[^a-z0-9]+', '-', suburb.lower()).strip('-')}-vic-{postcode}"


def file_stem(suburb, postcode):
    return f"{suburb.title().replace(' ', '_')}_{postcode}"


def snap_dir(snapshot):
    return RENT_DIR / snapshot


def search_dir(snapshot, worker=None):
    d = snap_dir(snapshot) / "search"
    return d / worker if worker else d


def parse_shard(text):
    """'2/4' -> (2, 4)"""
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text or "")
    if not m or not 1 <= int(m.group(1)) <= int(m.group(2)):
        sys.exit("--shard must look like 2/4 (your share / number of people)")
    return int(m.group(1)), int(m.group(2))


# ----------------------------------------------------------------------------
# Splitting the suburbs between people
# ----------------------------------------------------------------------------
def load_suburbs(path=SUBURBS):
    """Suburbs to search, biggest first. Sorted the same way on every computer."""
    s = pd.read_csv(path)
    s["suburb"] = s["suburb"].str.strip().str.upper()
    s = s.drop_duplicates(["suburb", "postcode"])
    return s.sort_values(["listing_count", "suburb", "postcode"],
                         ascending=[False, True, True]).reset_index(drop=True)


def make_plan(suburbs, workers):
    """Deal suburbs out in a 'snake' (1,2,3,4,4,3,2,1,...) so everyone gets a fair mix of
    big and small suburbs. The result is identical on every computer."""
    plan = suburbs.copy()
    rnd, pos = plan.index // workers, plan.index % workers
    plan["shard"] = [p + 1 if r % 2 == 0 else workers - p for r, p in zip(rnd, pos)]
    # Pages estimated from the 2025 Domain counts; rent.com.au often has more listings
    plan["est_pages"] = [min(DEFAULT_MAX_PAGES, max(1, math.ceil(c / PER_PAGE))) for c in plan["listing_count"]]
    return plan


# ----------------------------------------------------------------------------
# Polite fetching
# ----------------------------------------------------------------------------
class PoliteFetcher:
    def __init__(self, delay, contact):
        ua = f"Mozilla/5.0 (compatible; {UA_TOKEN}/1.0; University of Melbourne coursework"
        ua += f"; contact: {contact})" if contact else ")"
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": ua,
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-AU,en;q=0.9",
        })
        self.delay = delay
        self.last = 0.0
        self.robots = self._load_robots()

    def _load_robots(self):
        rp = RobotFileParser()
        try:
            r = self.session.get(BASE + "/robots.txt", timeout=30)
        except requests.RequestException as e:
            sys.exit(f"Could not reach rent.com.au ({e}). Check your internet connection.")
        if r.status_code in (401, 403, 429, 503):
            raise Blocked(f"robots.txt returned {r.status_code}")
        rp.parse(r.text.splitlines() if r.status_code == 200 else [])
        crawl_delay = rp.crawl_delay(UA_TOKEN)
        if crawl_delay and float(crawl_delay) > self.delay:
            print(f"robots.txt asks for {crawl_delay}s between requests; using that.")
            self.delay = float(crawl_delay)
        for path in ("/properties/box-hill-vic-3128", "/property/example-1"):
            if not rp.can_fetch(UA_TOKEN, BASE + path):
                raise Blocked(f"robots.txt does not allow {path.split('/')[1]} pages")
        return rp

    def get(self, url):
        """Return page HTML, or None for a 404. Raises Blocked if access is refused."""
        if not self.robots.can_fetch(UA_TOKEN, url):
            print(f"    skipped (robots.txt): {url}")
            return None
        for attempt in range(3):
            wait = self.delay * random.uniform(0.7, 1.3) - (time.time() - self.last)
            if wait > 0:
                time.sleep(wait)
            try:
                r = self.session.get(url, timeout=30)
            except requests.RequestException as e:
                self.last = time.time()
                print(f"    network error ({e.__class__.__name__}), retrying in 30s")
                time.sleep(30)
                continue
            self.last = time.time()

            if r.status_code == 404:
                return None
            if r.status_code == 429 and attempt == 0:
                pause = min(int(r.headers.get("Retry-After", "300") or 300), 900)
                print(f"    site asked us to slow down (429); pausing {pause}s and doubling the delay")
                self.delay *= 2
                time.sleep(pause)
                continue
            if r.status_code in (401, 403, 429, 503) or self._is_challenge(r.text):
                raise Blocked(f"HTTP {r.status_code} for {url}")
            if r.status_code >= 500:
                time.sleep(30 * (attempt + 1))
                continue
            r.raise_for_status()
            return r.text
        raise RuntimeError(f"gave up on {url}")

    @staticmethod
    def _is_challenge(html):
        head = html[:5000].lower()
        return any(s in head for s in (
            "<title>just a moment", "attention required! | cloudflare",
            "enable javascript and cookies to continue", "cf-chl-",
        ))


# ----------------------------------------------------------------------------
# The listing data each page embeds for its own JavaScript
# ----------------------------------------------------------------------------
NEXT_CHUNK = re.compile(r'self\.__next_f\.push\(\[1,"((?:[^"\\]|\\.)*)"\]\)')


def page_data(html):
    """rent.com.au pages (Next.js) carry their data as escaped text chunks; decode and join them."""
    parts = []
    for chunk in NEXT_CHUNK.findall(html):
        try:
            parts.append(json.loads('"' + chunk + '"'))
        except json.JSONDecodeError:
            continue
    return "".join(parts)


def listing_objects(html):
    """Every listing record in the page's embedded data: dicts that have an id and a latitude.
    From each "latitude" key, walk back to the '{' that opens the object holding it."""
    data = page_data(html)
    decoder = json.JSONDecoder()
    found = {}
    for m in re.finditer(r'"latitude"', data):
        pos = m.start()
        for _ in range(500):
            pos = data.rfind("{", 0, pos)
            if pos < 0:
                break
            try:
                obj, _ = decoder.raw_decode(data, pos)
            except ValueError:
                continue
            if isinstance(obj, dict) and "latitude" in obj and "id" in obj:
                found.setdefault(str(obj["id"]), obj)
                break
    return list(found.values())


def row_from_listing(o):
    """Search-page fields from one listing record. Whitelist only: the record also holds the
    agent's name and phone numbers (under "contact"), which are never copied."""
    walk = o.get("walkability_score") or {}
    company = o.get("company") or (o.get("contact") or {}).get("company") or {}
    lid = str(o.get("id"))
    street = squash(o.get("street_address")).rstrip(", ")
    suburb = squash(o.get("suburb")).upper() or None
    postcode = to_int(r"(\d{4})", str(o.get("postcode", "")))
    listed = o.get("activated_at")
    return {
        "listing_id": lid,
        "url": o.get("url") or f"{BASE}/property/{lid}",
        "address": f"{street}, {suburb.title()} VIC {postcode}" if street and suburb else street or None,
        "suburb": suburb,
        "postcode": postcode,
        "weekly_rent": to_float(o.get("weekly_rent_number")) or parse_weekly_rent(o.get("weekly_rent")),
        "price_text": squash(o.get("weekly_rent")).replace("$$", "$") or None,
        "bedrooms": o.get("bedrooms"),
        "bathrooms": o.get("bathrooms"),
        "carspaces": o.get("car_spaces"),
        "property_type": o.get("property_type"),
        "lat": to_float(o.get("latitude")),
        "lon": to_float(o.get("longitude")),
        "date_listed": str(listed)[:10] if listed else None,
        "available_date": parse_date(o.get("date_available")),
        "pets_allowed": o.get("pets_allowed"),
        "agency": company.get("trading_name"),
        "walk_score": walk.get("walk_score"),
        "transit_score": walk.get("transit_score"),
    }


def flatten_features(features):
    """{'heating': ['gas'], 'kitchen': ['dishwasher']} -> ['Heating: gas', 'Kitchen: dishwasher']"""
    items = []
    if isinstance(features, dict):
        for group, values in features.items():
            for v in values if isinstance(values, list) else [values]:
                items.append(f"{str(group).replace('_', ' ').capitalize()}: {v}")
    elif isinstance(features, list):
        items = [str(v) for v in features]
    return items


PHONE_OR_EMAIL = re.compile(r"(\+?61|0)[\d ()-]{8,14}\d|[\w.+-]+@[\w-]+\.[\w.]+")


def detail_from_listing(o, lid):
    feats = flatten_features(o.get("features"))
    if o.get("furnished"):
        feats.append("Furnished")
    elif o.get("partially_furnished"):
        feats.append("Partially furnished")
    if (o.get("ensuites") or 0) > 0:
        feats.append("Ensuite")
    photos = [g for g in (o.get("gallery") or []) if (g or {}).get("gallery_type", "photo") == "photo"]
    description = PHONE_OR_EMAIL.sub("[removed]", squash(o.get("description")))[:3000]
    return {
        "listing_id": lid,
        "bond": to_float(o.get("bond")),
        "weekly_rent_detail": to_float(o.get("weekly_rent_number")),
        "furnished": o.get("furnished"),
        "partially_furnished": o.get("partially_furnished"),
        "ensuites": o.get("ensuites"),
        "structured_features": ", ".join(dict.fromkeys(feats)) or None,
        "byline": squash(o.get("byline")) or None,
        "description": description or None,
        "photo_count": len(photos) or None,
        "detail_scraped_date": now(),
    }


# ----------------------------------------------------------------------------
# Backup parsers: read the visible page text if the embedded data ever disappears
# ----------------------------------------------------------------------------
PROPERTY_TYPES = [
    "Serviced Apartment", "Block of Units", "Semi-detached", "Semi-Detached", "Granny Flat",
    "Car Space", "Apartment", "Townhouse", "Penthouse", "Warehouse", "Retirement", "Acreage",
    "Cottage", "Duplex", "Terrace", "Studio", "House", "Villa", "Rural", "Other", "Unit",
    "Flat", "Loft", "Room", "Farm", "Land",
]
TYPE_RE = re.compile(r"\|\s*(" + "|".join(re.escape(t) for t in PROPERTY_TYPES) + r")\b")


def _listing_id(href):
    m = PROPERTY_HREF.search(href.split("?")[0].split("#")[0])
    return m.group(1) if m else None


def find_cards(soup):
    """Map listing id -> (url, element holding that listing's card)."""
    cards = {}
    for a in soup.find_all("a", href=True):
        lid = _listing_id(a["href"])
        if not lid:
            continue
        node = a
        for _ in range(6):  # climb while the parent still only holds this one listing
            parent = node.parent
            if parent is None or parent.name in ("body", "html", "[document]", "main"):
                break
            ids = {_listing_id(x["href"]) for x in parent.find_all("a", href=True)} - {None}
            if ids != {lid}:
                break
            node = parent
        if lid not in cards or len(node.get_text()) > len(cards[lid][1].get_text()):
            cards[lid] = (urljoin(BASE, a["href"].split("?")[0]), node)
    return cards


def parse_card(lid, url, node):
    text = squash(node.get_text(" ", strip=True))
    price_node = node.find(string=re.compile(r"\$\s?\d"))
    price_text = squash(price_node) if price_node else None
    addr_node = node.find(string=re.compile(r"\bVIC\s+\d{4}\s*$"))
    address = squash(addr_node) if addr_node else None
    m = re.search(r",\s*([^,]+?)\s+VIC\s+(\d{4})\s*$", address or "", re.I)
    ptype = TYPE_RE.search(text)
    return {
        "listing_id": lid, "url": url, "address": address,
        "suburb": m.group(1).strip().upper() if m else None,
        "postcode": int(m.group(2)) if m else None,
        "weekly_rent": parse_weekly_rent(price_text), "price_text": price_text,
        "bedrooms": to_int(r"(\d+)\s*beds?\b", text),
        "bathrooms": to_int(r"(\d+)\s*bath(?:room)?s?\b", text),
        "carspaces": to_int(r"(\d+)\s*car\s*spaces?\b", text),
        "property_type": ptype.group(1) if ptype else None,
        "pets_allowed": "pets allowed" in text.lower(),
    }


def detail_from_text(html, lid):
    soup = BeautifulSoup(html, "html.parser")
    features = []
    heading = soup.find(lambda t: t.name in ("h2", "h3") and "property features" in t.get_text().lower())
    if heading:
        for el in heading.find_all_next():
            if el.name == "h2" and el is not heading:
                break
            item = squash(el.get_text(" ", strip=True)) if el.name == "li" else ""
            if item and not re.match(r"^\d+\s+(bedroom|bathroom|garage|car)", item, re.I) and item not in features:
                features.append(item)
    text = squash(soup.get_text(" ", strip=True))
    bond = re.search(r"Bond\s*\$\s?([\d,]+)", text)
    return {"listing_id": lid, "bond": money(bond.group(1)) if bond else None,
            "structured_features": ", ".join(features) or None, "detail_scraped_date": now()}


# ----------------------------------------------------------------------------
# Page parsers
# ----------------------------------------------------------------------------
def parse_search_page(html):
    """Returns (rows, total listings the site reports, whether the page says 'no results')."""
    soup = BeautifulSoup(html, "html.parser")
    text = squash(soup.get_text(" ", strip=True))
    m = re.search(r"of\s+([\d,]+)\s+rental propert", text, re.I)
    total = int(m.group(1).replace(",", "")) if m else None
    no_results = bool(re.search(r"\b(no|0)\s+(rental\s+)?properties\b", text, re.I))
    rows, method = [row_from_listing(o) for o in listing_objects(html)], "page data"
    if not rows:
        rows = [parse_card(lid, url, node) for lid, (url, node) in find_cards(soup).items()]
        method = "visible text"
    for r in rows:
        r["parse_method"] = method
    return rows, total, no_results


def parse_detail_page(html, url):
    lid = _listing_id(url) or ""
    objs = listing_objects(html)
    # Only trust the record whose id matches this URL: the page can also carry
    # "similar listings", and their rent or bond must never be attached to this one.
    o = next((x for x in objs if str(x.get("id")) == lid), None) if lid else (objs[0] if objs else None)
    return detail_from_listing(o, lid or str(o.get("id"))) if o else detail_from_text(html, lid)


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def save_debug(name, html):
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    (DEBUG_DIR / name).write_text(html, encoding="utf-8")
    return DEBUG_DIR / name


def show_parsed(rows, detail):
    """Print what was parsed, without addresses or links."""
    keys = ["listing_id", "suburb", "weekly_rent", "bedrooms", "bathrooms", "carspaces",
            "property_type", "lat", "lon", "date_listed", "agency"]
    for r in rows[:3]:
        print("   ", {k: r.get(k) for k in keys})
    filled = pd.DataFrame(rows, columns=SEARCH_COLUMNS).notna().mean().mul(100).round(0)
    print("    % filled:", filled[filled.index.isin(keys + ["walk_score", "available_date"])].to_dict())
    if detail:
        print("\n    listing page:", {k: (v[:60] + "...") if isinstance(v, str) and len(v) > 60 else v
                                    for k, v in detail.items() if k != "description"})


def offline_check():
    """Parse the saved debug pages (no internet). Returns (search rows, detail dict)."""
    search_file, detail_file = DEBUG_DIR / "test_search.html", DEBUG_DIR / "test_detail.html"
    if not search_file.exists():
        sys.exit(f"No saved pages in {DEBUG_DIR}. Run the live test once: "
                 "python3 scripts/scrape_rentcomau.py test --contact you@...")
    rows, total, _ = parse_search_page(search_file.read_text(encoding="utf-8"))
    detail = None
    if detail_file.exists() and rows:
        detail = parse_detail_page(detail_file.read_text(encoding="utf-8"), rows[0]["url"])
    print(f"Saved search page: site says {total} listings; parsed {len(rows)} "
          f"(method: {rows[0]['parse_method'] if rows else 'none'})")
    return rows, detail


def run_test(fetch, args):
    slug = suburb_slug(args.suburb, args.postcode)
    url = f"{BASE}/properties/{slug}"
    print(f"1) Search page: {url}")
    html = fetch.get(url)
    if html is None:
        sys.exit("   Page not found (404). Check the suburb/postcode.")
    save_debug("test_search.html", html)
    rows, total, _ = parse_search_page(html)
    print(f"   site says {total} listings; parsed {len(rows)} on this page "
          f"(method: {rows[0]['parse_method'] if rows else 'none'})")
    if not rows:
        sys.exit("   No listings parsed. The page layout may have changed; tell the group.")
    print(f"\n2) Listing page for listing {rows[0]['listing_id']}")
    html = fetch.get(rows[0]["url"])
    detail = parse_detail_page(html, rows[0]["url"]) if html else None
    if html:
        save_debug("test_detail.html", html)
    show_parsed(rows, detail)
    print(f"\nTest passed. Saved pages are in {DEBUG_DIR} (they contain agents' details: don't share them).")


def run_plan(args):
    workers = args.workers or 4
    plan = make_plan(load_suburbs(args.suburbs), workers)
    out = snap_dir(args.snapshot)
    out.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out / "plan.csv", index=False)
    table = plan.groupby("shard").agg(suburbs=("suburb", "size"), est_pages=("est_pages", "sum"),
                                      domain_listings=("listing_count", "sum"))
    table["est_minutes"] = (table["est_pages"] * args.delay / 60).round(0).astype(int)
    print(table.to_string())
    print(f"\nEstimates use 2025 Domain counts; rent.com.au often lists more, so allow ~1.5x.")
    print(f"Plan saved to {out / 'plan.csv'}\n\nEach person runs (from the repo's top folder):")
    for shard in range(1, workers + 1):
        print(f"  person {shard}: python3 scripts/scrape_rentcomau.py search --shard {shard}/{workers} "
              f"--worker <name> --contact <email>")


def run_search(fetch, args):
    shard, workers = parse_shard(args.shard)
    plan = make_plan(load_suburbs(args.suburbs), workers)
    mine = plan[plan["shard"] == shard]
    if args.limit:
        mine = mine.head(args.limit)
    out = search_dir(args.snapshot, args.worker)
    out.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in search_dir(args.snapshot).glob("*/*.csv")}   # done by anyone
    todo = [(r.suburb, r.postcode) for r in mine.itertuples() if file_stem(r.suburb, r.postcode) not in done]
    print(f"Share {shard}/{workers} ({args.worker}): {len(mine)} suburbs, {len(mine) - len(todo)} already done, "
          f"{len(todo)} to go (one request every ~{fetch.delay:.0f}s)")

    failures_in_a_row = 0
    for i, (suburb, postcode) in enumerate(todo, 1):
        slug = suburb_slug(suburb, postcode)
        rows, seen = [], set()
        try:
            for page in range(1, args.max_pages + 1):
                url = f"{BASE}/properties/{slug}" + (f"/p{page}" if page > 1 else "")
                html = fetch.get(url)
                if html is None:
                    break
                page_rows, total, no_results = parse_search_page(html)
                new = [r for r in page_rows if r["listing_id"] not in seen]
                if not new:
                    if page == 1 and not no_results and total is None:
                        print(f"    nothing parsed on {url}; saved {save_debug(f'unparsed_{slug}.html', html)}")
                    break
                for r in new:
                    r["page"] = page
                seen.update(r["listing_id"] for r in new)
                rows.extend(new)
                if total is None or page * PER_PAGE >= total:
                    break
            failures_in_a_row = 0
        except Blocked as e:
            print(f"\nSTOPPED: the site refused access ({e}). Do not try to get around this. "
                  "Progress so far is saved; tell the group and use the course dataset.")
            return
        except Exception as e:
            failures_in_a_row += 1
            print(f"[{i}/{len(todo)}] {suburb} {postcode}: FAILED ({e}); will retry next run")
            if failures_in_a_row >= 5:
                print("\nSTOPPED after 5 failures in a row. Check your connection and rerun later.")
                return
            continue

        stamp = now()
        for r in rows:
            r.update(search_suburb=suburb, search_postcode=postcode, worker=args.worker, scraped_date=stamp,
                     in_searched_suburb=(r.get("suburb") == suburb and r.get("postcode") == postcode))
        pd.DataFrame(rows, columns=SEARCH_COLUMNS).to_csv(out / f"{file_stem(suburb, postcode)}.csv", index=False)
        inside = sum(r["in_searched_suburb"] for r in rows)
        print(f"[{i}/{len(todo)}] {suburb} {postcode}: {len(rows)} listings ({inside} in this suburb)")

    print("\nYour share is finished. Send your folder "
          f"{out.relative_to(config.ROOT)} to the person merging (privately, not via GitHub).")


def run_details(fetch, args):
    files = [f for f in sorted(search_dir(args.snapshot, args.worker).glob("*.csv")) if f.stat().st_size > 0]
    if not files:
        sys.exit(f"No search results for worker '{args.worker}' yet. Run the search stage first.")
    search = pd.concat([pd.read_csv(f, dtype={"listing_id": str}) for f in files])
    search = search[search["in_searched_suburb"].astype(str).str.lower() == "true"].drop_duplicates("listing_id")

    folder = snap_dir(args.snapshot) / "details"
    folder.mkdir(parents=True, exist_ok=True)
    out_file, fail_file = folder / f"{args.worker}.csv", folder / f"{args.worker}_failed.csv"
    done = set()
    for f in (out_file, fail_file):
        if f.exists() and f.stat().st_size > 0:
            done |= set(pd.read_csv(f, dtype={"listing_id": str}, usecols=["listing_id"])["listing_id"])
    todo = search[~search["listing_id"].isin(done)][["listing_id", "url"]].values.tolist()
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} listing pages already done, {len(todo)} to go "
          f"(about {len(todo) * fetch.delay / 3600:.1f} hours)")

    new_file = not out_file.exists() or out_file.stat().st_size == 0
    failures_in_a_row = 0
    with open(out_file, "a", newline="", encoding="utf-8") as fh, \
         open(fail_file, "a", newline="", encoding="utf-8") as fh_fail:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_COLUMNS, extrasaction="ignore")
        fail_writer = csv.writer(fh_fail)
        if new_file:
            writer.writeheader()
        if fail_file.stat().st_size == 0:
            fail_writer.writerow(["listing_id", "reason"])
        for i, (lid, url) in enumerate(todo, 1):
            try:
                html = fetch.get(url)
                if html is None:                      # leased and taken down since the search
                    fail_writer.writerow([lid, "gone (404)"])
                    continue
                row = parse_detail_page(html, url)
                row["listing_id"] = lid
                writer.writerow(row)
                failures_in_a_row = 0
            except Blocked as e:
                print(f"\nSTOPPED: the site refused access ({e}). Progress is saved.")
                return
            except Exception as e:
                failures_in_a_row += 1
                print(f"    {lid}: FAILED ({e})")
                if failures_in_a_row >= 5:
                    print("\nSTOPPED after 5 failures in a row. Progress is saved; rerun later.")
                    return
                continue
            if i % 25 == 0:
                fh.flush()
                fh_fail.flush()
                print(f"[{i}/{len(todo)}] listing pages done")
    print("\nDetails finished (or reached --limit). Send details/"
          f"{args.worker}.csv with your search folder.")


def status_table(snapshot=DEFAULT_SNAPSHOT, workers=None, suburbs_path=SUBURBS):
    """Progress per share: suburbs done and listings found so far."""
    plan_file = snap_dir(snapshot) / "plan.csv"
    if workers:
        plan = make_plan(load_suburbs(suburbs_path), workers)
    elif plan_file.exists():
        plan = pd.read_csv(plan_file)
    else:
        sys.exit("Give --workers N (or run the plan stage first).")
    files = {p.stem: p for p in search_dir(snapshot).glob("*/*.csv")}
    stems = [file_stem(s, p) for s, p in zip(plan["suburb"], plan["postcode"])]
    plan["done"] = [s in files for s in stems]
    plan["worker"] = [files[s].parent.name if s in files else None for s in stems]
    plan["listings"] = [len(pd.read_csv(files[s], usecols=["listing_id"])) if s in files else 0 for s in stems]
    table = plan.groupby("shard").agg(
        suburbs=("suburb", "size"), done=("done", "sum"), listings_found=("listings", "sum"),
        workers=("worker", lambda w: ", ".join(sorted(set(w.dropna()))) or "-"))
    table["percent_done"] = (100 * table["done"] / table["suburbs"]).round(0)
    details = {f.stem: len(pd.read_csv(f, usecols=["listing_id"]))
               for f in (snap_dir(snapshot) / "details").glob("*.csv") if not f.stem.endswith("_failed")}
    return table, details


def merge(snapshot=DEFAULT_SNAPSHOT, verbose=True):
    """Combine every person's search (and optional details) files into one CSV."""
    files = [f for f in sorted(search_dir(snapshot).glob("*/*.csv")) if f.stat().st_size > 0]
    if not files:
        print("Nothing to merge yet: no search files in", search_dir(snapshot).relative_to(config.ROOT))
        return None
    search = pd.concat([pd.read_csv(f, dtype={"listing_id": str}) for f in files], ignore_index=True)
    raw_rows = len(search)
    search["in_searched_suburb"] = search["in_searched_suburb"].astype(str).str.lower() == "true"
    # A listing can show up in several suburbs' results (and so in several people's files):
    # keep one copy, preferring the search of its own suburb, then the most recent.
    search = (search.sort_values(["in_searched_suburb", "scraped_date"], ascending=[False, False])
                    .drop_duplicates("listing_id"))
    df = search
    detail_files = [f for f in (snap_dir(snapshot) / "details").glob("*.csv")
                    if not f.stem.endswith("_failed") and f.stat().st_size > 0]
    if detail_files:
        details = (pd.concat([pd.read_csv(f, dtype={"listing_id": str}) for f in detail_files])
                     .drop_duplicates("listing_id", keep="last"))
        df = search.merge(details, on="listing_id", how="left")
        df["weekly_rent"] = df["weekly_rent_detail"].combine_first(df["weekly_rent"])
    listed = pd.to_datetime(df["date_listed"], errors="coerce")
    df["days_listed"] = (pd.to_datetime(df["scraped_date"], errors="coerce").dt.normalize() - listed).dt.days
    df["snapshot"] = snapshot
    df["source"] = "rent.com.au"
    df = df.reindex(columns=MERGED_COLUMNS)
    out = snap_dir(snapshot) / "vic_rentals_rentcomau.csv"
    df.to_csv(out, index=False)
    if verbose:
        print(f"Merged {len(files)} suburb files from {len({f.parent.name for f in files})} people: "
              f"{raw_rows:,} rows -> {len(df):,} unique listings")
        print(f"  in their searched suburb: {df['in_searched_suburb'].sum():,}")
        for col in ["weekly_rent", "lat", "date_listed", "bond", "structured_features"]:
            print(f"  with {col:20} {df[col].notna().mean():6.1%}")
        print(f"Saved to {out.relative_to(config.ROOT)} (read automatically by scripts/listings.py)")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["plan", "test", "search", "details", "status", "merge"])
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT, help="collection name, e.g. 2026-09 (same for everyone)")
    ap.add_argument("--suburbs", default=SUBURBS, help="CSV with suburb, postcode, listing_count")
    ap.add_argument("--workers", type=int, help="number of people sharing the work (plan, status)")
    ap.add_argument("--shard", help="your share, e.g. 2/4 (search)")
    ap.add_argument("--worker", help="your short name, e.g. kerri (search, details)")
    ap.add_argument("--contact", help="your uni email, shown to the site in the User-Agent")
    ap.add_argument("--delay", type=float, default=5.0, help="average seconds between requests (min 3)")
    ap.add_argument("--limit", type=int, help="only do this many suburbs (search) or listings (details)")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--suburb", default="Box Hill", help="suburb for the test")
    ap.add_argument("--postcode", default="3128", help="postcode for the test")
    ap.add_argument("--offline", action="store_true", help="test: parse the saved pages, no internet")
    args = ap.parse_args()

    if args.stage == "plan":
        return run_plan(args)
    if args.stage == "status":
        table, details = status_table(args.snapshot, args.workers, args.suburbs)
        print(table.to_string())
        print("details rows per person:", details or "none yet")
        return
    if args.stage == "merge":
        merge(args.snapshot)
        return
    if args.stage == "test" and args.offline:
        rows, detail = offline_check()
        show_parsed(rows, detail)
        return

    if args.stage in ("search", "details"):
        if not args.worker or not WORKER_NAME.match(args.worker):
            sys.exit("Add --worker with a short lowercase name, e.g. --worker kerri")
        if args.stage == "search" and not args.shard:
            sys.exit("Add --shard, e.g. --shard 2/4 (see the plan stage)")
    if not args.contact:
        sys.exit("Add --contact your.name@student.unimelb.edu.au so the site knows who is visiting.")
    if args.delay < 3:
        sys.exit("Please keep --delay at 3 seconds or more.")
    try:
        fetch = PoliteFetcher(args.delay, args.contact)
    except Blocked as e:
        sys.exit(f"STOPPED before scraping: {e}. The site is not allowing automated access, "
                 "so use the course dataset instead.")
    {"test": run_test, "search": run_search, "details": run_details}[args.stage](fetch, args)


if __name__ == "__main__":
    main()
