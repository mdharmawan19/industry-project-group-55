import pandas as pd

def load_dffh_suburb_panel(path):
    """Reshape the DFFH 'Moving annual median rent by suburb and town'
    workbook into a long panel.

    Returns: region, area, property_type, quarter, count, median_rent
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
        wide["property_type"] = sheet
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


import re

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