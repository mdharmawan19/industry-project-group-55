"""Population and housing growth for DFFH areas, and whether they explain rent growth (Question 2).

Population comes from notebook 04 (`sa2_features.csv`: ABS estimates 2021-25 and Victoria in Future
projections); dwellings from Victoria in Future (`vif.load_vif_sa2`). Both are by SA2. Each DFFH area is linked
to SA2s through its suburbs (DFFH crosswalk -> suburb_sa2_crosswalk, using each suburb's main SA2), and SA2
counts are summed before growth is computed, so a large SA2 counts more than a small one.
"""
import numpy as np
import pandas as pd

COUNTS = ["POP_2021", "POP_2025", "VIF_POP_2026", "VIF_POP_2031", "DWELL_2021", "DWELL_2026", "DWELL_2031"]


def _cagr(end, start, years):
    return ((end / start) ** (1 / years) - 1) * 100


def sa2_table(sa2_features, dwellings):
    """One row per SA2 with the population and dwelling counts used here."""
    return sa2_features.merge(dwellings, on="SA2_CODE21")[["SA2_CODE21", "SA2_NAME21", "GCCSA"] + COUNTS]


def main_sa2(suburb_sa2):
    """Each suburb's main SA2 (the one most of its listings fall in)."""
    return (suburb_sa2.sort_values("share", ascending=False)
            .drop_duplicates("suburb_key")[["suburb_key", "SA2_CODE21"]])


def growth_rates(t):
    """% a year from summed counts: actual population 2021-25, dwellings 2021-26, projections 2026-31."""
    return pd.DataFrame({
        "pop_growth_2021_25": _cagr(t["POP_2025"], t["POP_2021"], 4),
        "dwelling_growth_2021_26": _cagr(t["DWELL_2026"], t["DWELL_2021"], 5),
        "pop_growth_2026_31": _cagr(t["VIF_POP_2031"], t["VIF_POP_2026"], 5),
        "dwelling_growth_2026_31": _cagr(t["DWELL_2031"], t["DWELL_2026"], 5),
    }, index=t.index)


def area_growth(dffh_crosswalk, suburb_sa2, sa2):
    """Population and dwelling growth (% a year) for each DFFH area."""
    pairs = (dffh_crosswalk.merge(main_sa2(suburb_sa2), on="suburb_key")
             [["dffh_area", "SA2_CODE21"]].drop_duplicates())
    totals = pairs.merge(sa2, on="SA2_CODE21").groupby("dffh_area")[COUNTS].sum()
    return growth_rates(totals)


def rent_growth(panel, start="2020Q3", end="2025Q3", property_type="All properties"):
    """Actual rent growth per area, % a year, between two quarters."""
    w = panel[panel["property_type"] == property_type].pivot_table(index="quarter", columns="area",
                                                                   values="median_rent")
    s, e = pd.Period(start, "Q"), pd.Period(end, "Q")
    return _cagr(w.loc[e], w.loc[s], (e - s).n / 4).rename("rent_growth")


def trend_miss(bt, method, origin="2020Q3"):
    """How far actual rent ended above (+) or below (-) the trend forecast, % a year."""
    b = bt[(bt["origin"] == origin) & (bt["method"] == method)].set_index("area")
    return (np.log(b["actual_5y"] / b["forecast_5y"]) / 5 * 100).rename("trend_miss")


def _loo_error(X, y):
    """Leave-one-out mean absolute error of a linear fit (each area predicted from all the others)."""
    X = np.column_stack([np.ones(len(y)), X])
    errs = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        beta = np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]
        errs.append(abs(y[i] - X[i] @ beta))
    return float(np.mean(errs))


def evidence(growth, rent, miss):
    """(table, errors): rank correlations of population and dwelling growth with actual rent growth and with
    the trend forecast's misses; and the leave-one-out error of predicting those misses without and with them."""
    d = growth.join(rent).join(miss).dropna()
    drivers = {"Population growth 2021-25": "pop_growth_2021_25",
               "Dwelling growth 2021-26": "dwelling_growth_2021_26"}
    table = pd.DataFrame({
        "with rent growth 2020-25": {k: d[v].corr(d["rent_growth"], method="spearman") for k, v in drivers.items()},
        "with where the trend forecast missed": {k: d[v].corr(d["trend_miss"], method="spearman")
                                                 for k, v in drivers.items()},
    }).round(2)
    y = d["trend_miss"].to_numpy()
    errors = {"trend alone": _loo_error(np.empty((len(y), 0)), y),
              "trend + population + dwellings": _loo_error(d[list(drivers.values())].to_numpy(), y)}
    return table, errors, len(d)


def uncovered_sa2_growth(sa2, suburb_sa2, dffh_crosswalk, gccsa="2GMEL"):
    """Projected growth for SA2s that contain our listings' suburbs but no DFFH area (e.g. growth corridors)."""
    covered = set(main_sa2(suburb_sa2).merge(dffh_crosswalk, on="suburb_key")["SA2_CODE21"])
    with_listings = set(suburb_sa2["SA2_CODE21"])
    t = sa2[sa2["SA2_CODE21"].isin(with_listings - covered) & (sa2["GCCSA"] == gccsa)].set_index("SA2_NAME21")
    return growth_rates(t[COUNTS]).sort_values("pop_growth_2026_31", ascending=False)
