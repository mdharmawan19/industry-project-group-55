"""DFFH Rental Report: Victorian Government rent history (new leases, from bonds lodged).

Notebook 03 uses the "Moving annual median rent by suburb and town" workbook: a 12-month rolling
median for ~146 suburb groups ("areas"), one sheet per property type, with a Count and a Median
column per quarter. The council-level "Quarterly median rents by LGA" workbook has the same layout
and also loads with load_dffh_workbook, but is not used.
"""
import re

import pandas as pd

# Sheet names differ between the workbooks ("1br flat" vs "1 bedroom flat"); use one set of names
_TYPE_RE = re.compile(r"(\d)\s*(?:br|bedroom)\s*(flat|house)", re.I)


def standard_type(sheet):
    """'1br flat' / '1 bedroom flat' -> '1 bedroom flat'; 'All Properties' -> 'All properties'."""
    m = _TYPE_RE.search(sheet)
    if m:
        return f"{m.group(1)} bedroom {m.group(2).lower()}"
    return "All properties" if "all" in sheet.lower() else sheet.strip()


def load_dffh_workbook(path):
    """Reshape a DFFH Rental Report workbook (suburb or LGA) into a long panel.

    Returns: region, area, property_type, quarter, count, median_rent.
    `area` is the suburb group (suburb workbook) or the council (LGA workbook).
    """
    xl = pd.ExcelFile(path)
    frames = []

    for sheet in xl.sheet_names:
        raw = pd.read_excel(xl, sheet_name=sheet, header=None)

        # Row 1 = quarter labels (each repeated for Count/Median)
        # Row 2 = 'Count' / 'Median'
        # Row 3+ = data. Col 0 = region (sparse), col 1 = area name.
        quarters = raw.iloc[1, 2:].ffill().tolist()
        measures = raw.iloc[2, 2:].tolist()

        body   = raw.iloc[3:].copy()
        region = body.iloc[:, 0].ffill()
        area   = body.iloc[:, 1]
        values = body.iloc[:, 2:]

        values.columns = pd.MultiIndex.from_arrays(
            [quarters, measures], names=["quarter", "measure"])
        values = values.set_index(
            pd.MultiIndex.from_arrays([region, area], names=["region", "area"]))

        long = (values.stack(level=["quarter", "measure"], future_stack=True)
                      .rename("value").reset_index())
        wide = (long.pivot_table(index=["region", "area", "quarter"],
                                 columns="measure", values="value",
                                 aggfunc="first").reset_index())
        wide["property_type"] = standard_type(sheet)
        frames.append(wide)

    panel = pd.concat(frames, ignore_index=True)
    panel.columns.name = None
    panel = panel.rename(columns={"Count": "count", "Median": "median_rent"})

    # Drop subtotals and blank labels
    panel = panel[panel["area"].notna()]
    panel = panel[panel["area"].astype(str).str.strip() != "Group Total"]

    panel["quarter"] = pd.PeriodIndex(
        pd.to_datetime(panel["quarter"], format="%b %Y"), freq="Q")
    for c in ["count", "median_rent"]:
        panel[c] = pd.to_numeric(panel[c], errors="coerce")

    panel["area"]   = panel["area"].astype(str).str.strip()
    panel["region"] = panel["region"].astype(str).str.strip()

    return (panel[["region", "area", "property_type", "quarter",
                   "count", "median_rent"]]
            .sort_values(["area", "property_type", "quarter"])
            .reset_index(drop=True))


load_dffh_suburb_panel = load_dffh_workbook   # original name, kept for existing code

# DFFH names that don't match the Domain suburb list
AREA_ALIASES = {
    "CBD": "Melbourne",
    "St Kilda Rd": "Melbourne",
    "West St Kilda": "St Kilda West",
    "East St Kilda": "St Kilda East",
    "East Brunswick": "Brunswick East",
    "West Brunswick": "Brunswick West",
    "East Hawthorn": "Hawthorn East",
    "Mt Eliza": "Mount Eliza",
    "Mt Martha": "Mount Martha",
    "Newcombe": "Newcomb",        # typo in DFFH source
    "Wanagaratta": "Wangaratta",  # typo in DFFH source
}

def area_to_suburbs(areas):
    """Explode DFFH area names into constituent suburbs.
    Returns long df: dffh_area, suburb (one row per suburb)."""
    rows = []
    for a in areas:
        for part in a.split("-"):
            part = part.strip()
            rows.append({"dffh_area": a,
                         "suburb": AREA_ALIASES.get(part, part)})
    return pd.DataFrame(rows).drop_duplicates()


def make_suburb_key(suburb, postcode):
    s = re.sub(r"\s+", "_", str(suburb).strip().upper())
    return f"{s}_{int(postcode)}"

def coverage_filter(panel, start="2015Q3", end="2025Q3",
                    min_coverage=0.95, min_count=20):
    """Keep area x property_type series with enough data over the window."""
    w = panel[(panel.quarter >= pd.Period(start, "Q")) &
              (panel.quarter <= pd.Period(end, "Q"))]
    stats = w.groupby(["area", "property_type"]).agg(
        cov=("median_rent", lambda x: x.notna().mean()),
        med_count=("count", "median"))
    keep = stats[(stats["cov"] >= min_coverage) &
                 (stats["med_count"] >= min_count)].index
    idx = w.set_index(["area", "property_type"])
    return idx.loc[idx.index.isin(keep)].reset_index()


def growth_cagr(panel, property_type="All properties",
                start="2015Q3", end="2025Q3"):
    """Annualised growth in median rent between two quarters, per area."""
    yrs = (pd.Period(end, "Q") - pd.Period(start, "Q")).n / 4
    sub = panel[(panel.property_type == property_type) &
                (panel.quarter.isin([pd.Period(start, "Q"),
                                     pd.Period(end, "Q")]))]
    piv = sub.pivot_table(index="area", columns="quarter", values="median_rent")
    piv.columns = ["rent_start", "rent_end"]
    piv = piv.dropna()
    piv["cagr_pct"] = ((piv.rent_end / piv.rent_start) ** (1 / yrs) - 1) * 100
    return piv.sort_values("cagr_pct", ascending=False).reset_index()    
def yoy_growth(panel, property_type="All properties", min_count=20):
    """Year-on-year growth per area per quarter.

    The DFFH series is a MOVING ANNUAL median: consecutive quarters share
    three of four quarters of underlying bonds. Differencing adjacent
    quarters therefore yields heavily autocorrelated, artificially smooth
    changes. Comparing the same quarter one year apart (lag 4) gives
    non-overlapping windows.
    """
    s = panel[panel.property_type == property_type].copy()
    s = s[s["count"] >= min_count]
    s = s.sort_values(["area", "quarter"])
    s["rent_lag4"] = s.groupby("area")["median_rent"].shift(4)
    s["quarter_lag4"] = s.groupby("area")["quarter"].shift(4)
    ok = (s["quarter"].astype("period[Q]") - s["quarter_lag4"].astype("period[Q]")
          ).apply(lambda d: getattr(d, "n", None)) == 4
    s = s[ok]
    s["yoy_pct"] = (s["median_rent"] / s["rent_lag4"] - 1) * 100
    return s[["region", "area", "quarter", "median_rent",
              "count", "yoy_pct"]].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Additions for the cleaned pipeline (notebook 03)
# --------------------------------------------------------------------------------------
def build_area_suburb_crosswalk(areas, postcodes, listing_keys=()):
    """DFFH area -> the suburbs it contains, each with the postcode our listings use.

    Many suburb names have more than one postcode (Wodonga: 3689 and 3690). The postcode is
    taken from our listings when that suburb appears there (either year), otherwise the
    first postcode in postcodes.csv.

    Returns: dffh_area, suburb, suburb_u, postcode, suburb_key, postcode_source
    """
    xw = area_to_suburbs(areas)
    xw["suburb_u"] = xw["suburb"].str.upper().str.strip()
    pc = postcodes.assign(suburb_u=postcodes["suburb"].str.upper().str.strip())
    candidates = pc.groupby("suburb_u")["postcode"].apply(list).to_dict()
    listing_keys = set(listing_keys)

    rows = []
    for r in xw.itertuples():
        options = candidates.get(r.suburb_u, [])
        in_listings = [p for p in options if make_suburb_key(r.suburb, p) in listing_keys]
        if in_listings:
            postcode, source = in_listings[0], "listings"
        elif options:
            postcode, source = options[0], "postcodes.csv"
        else:
            postcode, source = None, "not found"
        rows.append({"dffh_area": r.dffh_area, "suburb": r.suburb, "suburb_u": r.suburb_u,
                     "postcode": postcode, "postcode_source": source,
                     "suburb_key": make_suburb_key(r.suburb, postcode) if postcode is not None else None})
    out = pd.DataFrame(rows)
    out["postcode"] = out["postcode"].astype("Int64")
    return out[["dffh_area", "suburb", "suburb_u", "postcode", "suburb_key", "postcode_source"]]


def count_weighted_rent(panel, property_type="All properties"):
    """One series per quarter: area medians averaged with each area's number of new leases as
    the weight (a mean of medians would give a tiny area the same say as a big one)."""
    s = panel[(panel["property_type"] == property_type) & panel["median_rent"].notna() & panel["count"].notna()]
    return s.groupby("quarter").apply(lambda g: (g["median_rent"] * g["count"]).sum() / g["count"].sum(),
                                      include_groups=False)


# Fixes found by checking the crosswalk against both years of listings
AREA_ALIASES["Ballarat"] = "Ballarat Central"   # DFFH's "Ballarat" area is the town centre
MISSING_POSTCODES = pd.DataFrame({"suburb": ["Ballarat Central", "Mildura"],   # absent from postcodes.csv
                                  "postcode": [3350, 3500]})
