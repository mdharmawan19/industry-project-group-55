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