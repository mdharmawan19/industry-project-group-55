#!/usr/bin/env python3
"""
Polite scraper for Victorian rental listings on rent.com.au (MAST30034 Project 2).

It runs in two stages:
  1. search   Suburb search pages (25 listings per page). Gives rent, beds, baths,
              car spaces, property type, address. About 970 pages for our 656 suburbs.
  2. details  One page per listing. Adds bond, available date, features, walk and
              transit scores, agency, description and (if the page has them) lat/lon.
              About 12,000 pages, so run it overnight; it picks up where it stopped.

Being a good guest on the site:
  - Reads robots.txt first and never requests a page it disallows.
  - Waits about 5 seconds between requests (random 3.5 to 6.5 s), longer if
    robots.txt asks for it.
  - Says who it is in the User-Agent (a student project, with your email if given).
  - Stops straight away if the site refuses access (403 / 429 / 503 / "checking your
    browser" page). It does not try to get around blocking.
  - Does not collect agent (people's) names, in line with the site's terms.

Commands (run from the repo's top folder, industry-project-group-55/):
  python3 scripts/scrape_rentcomau.py test             # 2 requests, shows what gets parsed
  python3 scripts/scrape_rentcomau.py search           # stage 1 (about 80 minutes)
  python3 scripts/scrape_rentcomau.py details          # stage 2 (overnight, resumable)
  python3 scripts/scrape_rentcomau.py merge            # combine into one CSV

Common options:  --contact you@student.unimelb.edu.au   --delay 5   --limit N
Input:           data/raw/domain/Data/suburb_summary.csv (course dataset)
Output folder:   data/raw/rentcomau/  (ignored by the repo's .gitignore)

Data source: rent.com.au, collected for MAST30034 coursework only. Do not share the
raw data or commit it to GitHub.
If you use this code, attribute it in the repo README.
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

BASE = "https://www.rent.com.au"
PER_PAGE = 25
DEFAULT_MAX_PAGES = 12            # 12 x 25 = 300 per suburb, same cap as the course dataset
UA_TOKEN = "MAST30034-student-project"
OUT = Path("data/raw/rentcomau")
PROPERTY_HREF = re.compile(r"/property/[^/?#]*?-(\d+)/?$")

SEARCH_COLUMNS = [
    "listing_id", "url", "address", "suburb", "postcode", "weekly_rent", "price_text",
    "bedrooms", "bathrooms", "carspaces", "property_type", "walk_label", "pets_allowed",
    "search_suburb", "search_postcode", "in_searched_suburb", "scraped_date",
]
DETAIL_COLUMNS = [
    "listing_id", "url", "address", "weekly_rent", "price_text", "bond", "available_date",
    "bedrooms", "bathrooms", "carspaces", "property_type", "structured_features",
    "walk_score", "walk_label", "transit_score", "transit_label", "agency",
    "photo_count", "lat", "lon", "description", "detail_scraped_date",
]
PROPERTY_TYPES = [
    "Serviced Apartment", "Block of Units", "Semi-detached", "Semi-Detached", "Granny Flat",
    "Car Space", "Apartment", "Townhouse", "Penthouse", "Warehouse", "Retirement", "Acreage",
    "Cottage", "Duplex", "Terrace", "Studio", "House", "Villa", "Rural", "Other", "Unit",
    "Flat", "Loft", "Room", "Farm", "Land",
]
TYPE_RE = re.compile(r"\|\s*(" + "|".join(re.escape(t) for t in PROPERTY_TYPES) + r")\b")
WALK_RE = re.compile(r"(Walker.s paradise|Very walkable|Somewhat walkable|Car[- ]dependent)", re.I)


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
class Blocked(Exception):
    pass


def squash(text):
    return " ".join(str(text).split()) if text is not None else ""


def to_int(pattern, text):
    m = re.search(pattern, text, re.I)
    return int(m.group(1)) if m else None


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
    if not text:
        return None
    if text.lower() == "now":
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
# Search page parsing
# ----------------------------------------------------------------------------
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


def split_address(address):
    m = re.search(r",\s*([^,]+?)\s+VIC\s+(\d{4})\s*$", address or "", re.I)
    return (m.group(1).strip().upper(), int(m.group(2))) if m else (None, None)


def parse_card(lid, url, node):
    text = squash(node.get_text(" ", strip=True))
    alts = [img.get("alt", "") for img in node.find_all("img")]

    price_node = node.find(string=re.compile(r"\$\s?\d"))
    price_text = squash(price_node) if price_node else None
    if price_text and len(price_text) < 4 and price_node.parent is not None:
        price_text = squash(price_node.parent.get_text(" ", strip=True))

    addr_node = node.find(string=re.compile(r"\bVIC\s+\d{4}\s*$"))
    address = squash(addr_node) if addr_node else None
    prop_type = None
    for alt in alts:  # e.g. "10 Merton Street, , Box Hill 3128, VIC House Photo"
        m = re.match(r"^(.*?),\s*,?\s*([^,]+?)\s+(\d{4}),\s*VIC\s+(.+?)\s+Photo$", alt)
        if m:
            prop_type = m.group(4)
            if not address:
                address = f"{m.group(1)}, {m.group(2)} VIC {m.group(3)}"
            break
    if not prop_type and (m := TYPE_RE.search(text)):
        prop_type = m.group(1)

    suburb, postcode = split_address(address)
    walk = WALK_RE.search(text)
    return {
        "listing_id": lid,
        "url": url,
        "address": address,
        "suburb": suburb,
        "postcode": postcode,
        "weekly_rent": parse_weekly_rent(price_text),
        "price_text": price_text,
        "bedrooms": to_int(r"(\d+)\s*beds?\b", text),
        "bathrooms": to_int(r"(\d+)\s*bath(?:room)?s?\b", text),
        "carspaces": to_int(r"(\d+)\s*car\s*spaces?\b", text),
        "property_type": prop_type,
        "walk_label": walk.group(1) if walk else None,
        "pets_allowed": any("pets allowed" in a.lower() for a in alts) or "pets allowed" in text.lower(),
    }


def parse_search_page(html):
    soup = BeautifulSoup(html, "html.parser")
    page_text = squash(soup.get_text(" ", strip=True))
    m = re.search(r"of\s+([\d,]+)\s+rental propert", page_text, re.I)
    total = int(m.group(1).replace(",", "")) if m else None
    no_results = bool(re.search(r"\b(no|0)\s+(rental\s+)?properties\b", page_text, re.I))
    rows = [parse_card(lid, url, node) for lid, (url, node) in find_cards(soup).items()]
    return rows, total, no_results


# ----------------------------------------------------------------------------
# Listing page parsing
# ----------------------------------------------------------------------------
def _find_coords_in_json(obj):
    """Search nested JSON for a lat/lon pair inside Victoria."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            lat = next((cur[k] for k in ("latitude", "lat") if k in cur), None)
            lon = next((cur[k] for k in ("longitude", "lng", "lon") if k in cur), None)
            try:
                lat, lon = float(lat), float(lon)
                if -39.3 < lat < -33.9 and 140.9 < lon < 150.1:
                    return lat, lon
            except (TypeError, ValueError):
                pass
            stack.extend(cur.values())
        elif isinstance(cur, list):
            stack.extend(cur)
    return None, None


def find_coords(soup, html):
    for tag in soup.find_all("script", type="application/ld+json") + soup.find_all("script", id="__NEXT_DATA__"):
        try:
            lat, lon = _find_coords_in_json(json.loads(tag.string or ""))
            if lat is not None:
                return lat, lon
        except (json.JSONDecodeError, TypeError):
            continue
    lat_pat = r"-3[4-9]\.\d{3,}"
    lon_pat = r"1(?:4\d|50)\.\d{3,}"
    for pat in (
        rf'\\?"lat(?:itude)?\\?"\s*:\s*\\?"?(-3[4-9]\.\d{{2,}})[\s\S]{{0,200}}?\\?"(?:lng|lon|longitude)\\?"\s*:\s*\\?"?(1(?:4\d|50)\.\d{{2,}})',
        rf'(?:center|q|ll|markers)=(?:[^&"]*?\|)?({lat_pat})(?:,|%2C)\s?({lon_pat})',
        rf'\bLatLng\(\s*({lat_pat})\s*,\s*({lon_pat})',
    ):
        if m := re.search(pat, html):
            return float(m.group(1)), float(m.group(2))
    return None, None


def parse_detail_page(html, url):
    soup = BeautifulSoup(html, "html.parser")
    lat, lon = find_coords(soup, html)
    photo_nums = [int(m.group(1)) for img in soup.find_all("img")
                  if (m := re.match(r"Property photo (\d+)", img.get("alt", "")))]

    # Features: list items under the "Property features" heading, up to the next h2
    features = []
    heading = soup.find(lambda t: t.name in ("h2", "h3") and "property features" in t.get_text().lower())
    if heading:
        for el in heading.find_all_next():
            if el.name == "h2" and el is not heading:
                break
            if el.name == "li":
                item = squash(el.get_text(" ", strip=True))
                if item and item not in features:
                    features.append(item)

    for t in soup(["script", "style", "noscript", "head"]):
        t.decompose()
    lines = [squash(x) for x in soup.get_text("\n").split("\n") if squash(x)]
    text = " ".join(lines)

    h1 = soup.find("h1")
    address = squash(h1.get_text(" ", strip=True)) if h1 else None
    start = text.find(address) if address else -1
    end = text.find("Property features")
    summary = text[start: end if end > start else start + 800] if start >= 0 else text[:3000]

    price_text = None
    for s in soup.find_all(string=re.compile(r"\$\s?[\d,]{2,}")):
        s_clean = squash(s)
        if not s_clean.lower().startswith("bond"):
            price_text = s_clean
            break
    bond = re.search(r"Bond\s*\$\s?([\d,]+)", text)
    avail = re.search(r"Available\s+(now|\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+\d{4})", summary, re.I)
    ptype = TYPE_RE.search(summary)
    walk = re.search(r"(\d{1,3})\s+(Walker.s paradise|Very walkable|Somewhat walkable|Car[- ]dependent)", text, re.I)
    transit = re.search(r"(\d{1,3})\s+(Rider.s paradise|Excellent|Good|Some|Minimal)\s+Transit", text, re.I)
    rent_id = re.search(r"Rent ID:?\s*(\d+)", text)

    desc = None
    if rent_id:  # description sits between the "Available ..." line and "Rent ID"
        if avail:
            desc_start = text.find(avail.group(0)) + len(avail.group(0))
        elif ptype:
            desc_start = text.find(ptype.group(0), max(start, 0)) + len(ptype.group(0))
        else:
            desc_start = max(start, 0)
        desc = text[desc_start: text.find(rent_id.group(0))].strip()[:3000] or None

    # Agency: the line just before the "Apply" button in the contact box (after the walk score)
    agency = None
    anchor = next((i for i, l in enumerate(lines) if l.lower().startswith("walkability score")), None)
    if anchor is not None:
        for i in range(anchor, len(lines)):
            if lines[i] == "Apply" and i > 0:
                candidate = lines[i - 1]
                if candidate.lower() not in ("enquire", "book inspection") and len(candidate) < 80:
                    agency = candidate
                break

    return {
        "listing_id": rent_id.group(1) if rent_id else _listing_id(url),
        "url": url,
        "address": address,
        "weekly_rent": parse_weekly_rent(price_text),
        "price_text": price_text,
        "bond": money(bond.group(1)) if bond else None,
        "available_date": parse_date(avail.group(1)) if avail else None,
        "bedrooms": to_int(r"(\d+)\s*beds?\b", summary),
        "bathrooms": to_int(r"(\d+)\s*bath(?:room)?s?\b", summary),
        "carspaces": to_int(r"(\d+)\s*car\s*spaces?\b", summary),
        "property_type": ptype.group(1) if ptype else None,
        "structured_features": ", ".join(features) or None,
        "walk_score": int(walk.group(1)) if walk else None,
        "walk_label": walk.group(2) if walk else None,
        "transit_score": int(transit.group(1)) if transit else None,
        "transit_label": transit.group(2) if transit else None,
        "agency": agency,
        "photo_count": max(photo_nums) if photo_nums else None,
        "lat": lat,
        "lon": lon,
        "description": desc,
        "detail_scraped_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ----------------------------------------------------------------------------
# Stages
# ----------------------------------------------------------------------------
def save_debug(name, html):
    d = OUT / "debug"
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(html, encoding="utf-8")
    return d / name


def run_test(fetch, args):
    slug = suburb_slug(args.suburb, args.postcode)
    url = f"{BASE}/properties/{slug}"
    print(f"1) Search page: {url}")
    html = fetch.get(url)
    if html is None:
        sys.exit("   Page not found (404). Check the suburb/postcode.")
    print(f"   saved to {save_debug('test_search.html', html)}")
    rows, total, _ = parse_search_page(html)
    print(f"   site says {total} listings; parsed {len(rows)} on this page")
    for r in rows[:3]:
        print("   ", {k: r[k] for k in ("listing_id", "address", "weekly_rent", "bedrooms",
                                          "bathrooms", "carspaces", "property_type")})
    if not rows:
        sys.exit("   No listings parsed. Send the saved HTML file's first lines to your teammate/Claude.")

    print(f"\n2) Listing page: {rows[0]['url']}")
    html = fetch.get(rows[0]["url"])
    if html is None:
        sys.exit("   That listing page was not found (it may have just been leased). Run the test again.")
    print(f"   saved to {save_debug('test_detail.html', html)}")
    d = parse_detail_page(html, rows[0]["url"])
    for k, v in d.items():
        v = (v[:80] + "...") if isinstance(v, str) and len(v) > 80 else v
        print(f"   {k:20} {v}")
    if d["lat"] is None:
        print("\n   Note: no coordinates found on the listing page. We will add them from the"
              " address later (geocoding) instead.")

    summary = pd.read_csv(args.summary)
    pages = sum(min(args.max_pages, max(1, math.ceil(c / PER_PAGE))) for c in summary["listing_count"])
    print(f"\nEstimated stage 1: ~{pages} pages, about {pages * fetch.delay / 60:.0f} minutes.")
    print(f"Estimated stage 2: ~{summary['listing_count'].sum():,} pages, about "
          f"{summary['listing_count'].sum() * fetch.delay / 3600:.0f} hours (run overnight, resumable).")


def run_search(fetch, args):
    folder = OUT / "search"
    folder.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.summary).sort_values("listing_count", ascending=False)
    if args.limit:
        summary = summary.head(args.limit)
    todo = [(r.suburb, r.postcode) for r in summary.itertuples()
            if not (folder / f"{file_stem(r.suburb, r.postcode)}.csv").exists()]
    print(f"{len(summary) - len(todo)} suburbs already done, {len(todo)} to go "
          f"(one request every ~{fetch.delay:.0f}s)")

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
                        path = save_debug(f"unparsed_{slug}.html", html)
                        print(f"    nothing parsed on {url}; saved {path}")
                    break
                seen.update(r["listing_id"] for r in new)
                rows.extend(new)
                if total is None or page * PER_PAGE >= total:
                    break
            failures_in_a_row = 0
        except Blocked as e:
            print(f"\nSTOPPED: the site refused access ({e}). Do not try to get around this. "
                  "Use the course dataset instead.")
            return
        except Exception as e:
            failures_in_a_row += 1
            print(f"[{i}/{len(todo)}] {suburb} {postcode}: FAILED ({e}); will retry next run")
            if failures_in_a_row >= 5:
                print("\nSTOPPED after 5 failures in a row. Check your connection and rerun later.")
                return
            continue

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in rows:
            r.update(search_suburb=suburb, search_postcode=postcode, scraped_date=now,
                     in_searched_suburb=(r["suburb"] == suburb.upper() and r["postcode"] == postcode))
        pd.DataFrame(rows, columns=SEARCH_COLUMNS).to_csv(folder / f"{file_stem(suburb, postcode)}.csv", index=False)
        inside = sum(r["in_searched_suburb"] for r in rows)
        print(f"[{i}/{len(todo)}] {suburb} {postcode}: {len(rows)} listings ({inside} in this suburb)")

    print("\nStage 1 finished. Next: python3 scripts/scrape_rentcomau.py details")


def run_details(fetch, args):
    files = sorted((OUT / "search").glob("*.csv"))
    if not files:
        sys.exit("Run the search stage first.")
    search = pd.concat([pd.read_csv(f, dtype={"listing_id": str}) for f in files if f.stat().st_size > 0])
    search = search.drop_duplicates("listing_id")

    out_file = OUT / "details.csv"
    fail_file = OUT / "details_failed.csv"
    done = set()
    if out_file.exists():
        done |= set(pd.read_csv(out_file, dtype={"listing_id": str}, usecols=["listing_id"])["listing_id"])
    if fail_file.exists():
        done |= set(pd.read_csv(fail_file, dtype={"listing_id": str})["listing_id"])

    todo = search[~search["listing_id"].isin(done)][["listing_id", "url"]].values.tolist()
    if args.limit:
        todo = todo[: args.limit]
    hours = len(todo) * fetch.delay / 3600
    print(f"{len(done)} listing pages already done, {len(todo)} to go (about {hours:.1f} hours)")

    new_file = not out_file.exists()
    failures_in_a_row = 0
    with open(out_file, "a", newline="", encoding="utf-8") as fh, \
         open(fail_file, "a", newline="", encoding="utf-8") as fh_fail:
        writer = csv.DictWriter(fh, fieldnames=DETAIL_COLUMNS)
        fail_writer = csv.writer(fh_fail)
        if new_file:
            writer.writeheader()
        if fail_file.stat().st_size == 0:
            fail_writer.writerow(["listing_id", "reason"])

        for i, (lid, url) in enumerate(todo, 1):
            try:
                html = fetch.get(url)
                if html is None:  # listing was taken down since stage 1
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

    print("\nStage 2 finished (or reached --limit). Next: python3 scripts/scrape_rentcomau.py merge")


def run_merge(args):
    files = [f for f in sorted((OUT / "search").glob("*.csv")) if f.stat().st_size > 0]
    if not files:
        sys.exit("Nothing to merge yet. Run the search stage first.")
    search = pd.concat([pd.read_csv(f, dtype={"listing_id": str}) for f in files if f.stat().st_size > 0])
    # A listing can show up in several suburb searches; keep the copy from its own suburb
    search = (search.sort_values("in_searched_suburb", ascending=False)
                    .drop_duplicates("listing_id"))
    df = search
    details_file = OUT / "details.csv"
    if details_file.exists():
        details = pd.read_csv(details_file, dtype={"listing_id": str}).drop_duplicates("listing_id", keep="last")
        df = search.merge(details, on="listing_id", how="left", suffixes=("", "_detail"))
        for col in ("address", "weekly_rent", "price_text", "bedrooms", "bathrooms",
                    "carspaces", "property_type", "walk_label", "url"):
            if f"{col}_detail" in df:
                df[col] = df[f"{col}_detail"].combine_first(df[col])
                df = df.drop(columns=f"{col}_detail")
    df["snapshot"] = datetime.now().strftime("%Y-%m")
    df["source"] = "rent.com.au"
    df.to_csv(OUT / "vic_rentals_rentcomau.csv", index=False)
    print(f"Saved {len(df):,} unique listings to {OUT / 'vic_rentals_rentcomau.csv'}")
    print(f"  in their searched suburb: {df['in_searched_suburb'].sum():,}")
    print(f"  with rent: {df['weekly_rent'].notna().sum():,}")
    if "lat" in df:
        print(f"  with listing-page details: {df['detail_scraped_date'].notna().sum():,} "
              f"| with coordinates: {df['lat'].notna().sum():,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["test", "search", "details", "merge"])
    ap.add_argument("--summary", default="data/raw/domain/Data/suburb_summary.csv",
                    help="course suburb_summary.csv (which suburbs to search)")
    ap.add_argument("--delay", type=float, default=5.0, help="average seconds between requests (min 3)")
    ap.add_argument("--contact", help="your uni email, shown to the site in the User-Agent")
    ap.add_argument("--limit", type=int, help="only do this many suburbs (search) or listings (details)")
    ap.add_argument("--max-pages", type=int, default=DEFAULT_MAX_PAGES)
    ap.add_argument("--suburb", default="Box Hill", help="suburb for the test")
    ap.add_argument("--postcode", default="3128", help="postcode for the test")
    args = ap.parse_args()

    if args.stage == "merge":
        run_merge(args)
        return
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
