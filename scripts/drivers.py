"""Population growth for DFFH areas, and whether it improves the rent forecasts (Question 2).

Population comes from notebook 04 (`sa2_features.csv`: ABS estimates 2021-25 and Victoria in Future projections),
by SA2. Each DFFH area is linked to SA2s through its suburbs (DFFH crosswalk -> suburb_sa2_crosswalk, using each
suburb's main SA2), and SA2 populations are summed before growth is computed, so a large SA2 counts more than a small
one.
"""
import numpy as np
import pandas as pd

COUNTS = ["POP_2021", "POP_2025", "VIF_POP_2026", "VIF_POP_2031"]


def _cagr(end, start, years):
    return ((end / start) ** (1 / years) - 1) * 100


def main_sa2(suburb_sa2):
    """Each suburb's main SA2 (the one most of its listings fall in)."""
    return (suburb_sa2.sort_values("share", ascending=False)
            .drop_duplicates("suburb_key")[["suburb_key", "SA2_CODE21"]])


def area_growth(dffh_crosswalk, suburb_sa2, sa2_features):
    """Population growth (% a year) for each DFFH area: actual 2021-25 and projected 2026-31."""
    pairs = (dffh_crosswalk.merge(main_sa2(suburb_sa2), on="suburb_key")
             [["dffh_area", "SA2_CODE21"]].drop_duplicates())
    t = pairs.merge(sa2_features[["SA2_CODE21"] + COUNTS], on="SA2_CODE21").groupby("dffh_area")[COUNTS].sum()
    return pd.DataFrame({"pop_growth_2021_25": _cagr(t["POP_2025"], t["POP_2021"], 4),
                         "pop_growth_2026_31": _cagr(t["VIF_POP_2031"], t["VIF_POP_2026"], 5)}, index=t.index)


def rent_growth(panel, start="2020Q3", end="2025Q3", property_type="All properties"):
    """Actual rent growth per area, % a year, between two quarters."""
    w = panel[panel["property_type"] == property_type].pivot_table(index="quarter", columns="area",
                                                                   values="median_rent")
    s, e = pd.Period(start, "Q"), pd.Period(end, "Q")
    return _cagr(w.loc[e], w.loc[s], (e - s).n / 4).rename("rent_growth")


def trend_miss(bt, method, origin="2020Q3"):
    """How far actual rent ended above (+) or below (-) a method's forecast, % a year."""
    b = bt[(bt["origin"] == origin) & (bt["method"] == method)].set_index("area")
    return (np.log(b["actual_5y"] / b["forecast_5y"]) / 5 * 100).rename("trend_miss")


def _loo_nudge(x, y):
    """Leave-one-out nudge for each area: fit y = b·(x − average) on all the other areas, then apply it to this one.
    x is centred on the average area, so a typical area gets no nudge: any gain comes from telling areas apart,
    not from shifting every forecast up or down."""
    nudge = np.empty(len(y))
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        xc = x[keep] - x[keep].mean()
        b = (xc @ y[keep]) / (xc @ xc)
        nudge[i] = (x[i] - x[keep].mean()) * b
    return nudge


def with_without(bt, growth, origin="2020Q3"):
    """Back-test error (%, 5 years ahead) of every method at one test date, without and with a population nudge.
    The nudge is learned from the other areas (leave-one-out) and uses each area's actual 2021-25 population growth,
    i.e. hindsight, which favours population. Returns (table, number of areas)."""
    rows = {}
    for method in bt["method"].unique():
        b = bt[(bt["origin"] == origin) & (bt["method"] == method)].set_index("area")
        d = b.join(growth["pop_growth_2021_25"]).join(trend_miss(bt, method, origin)).dropna()
        nudge = _loo_nudge(d["pop_growth_2021_25"].to_numpy(), d["trend_miss"].to_numpy())
        adjusted = d["forecast_5y"] * np.exp(5 * nudge / 100)
        rows[method] = {"without": d["abs_pct_error"].mean(),
                        "with": (np.abs(adjusted / d["actual_5y"] - 1) * 100).mean()}
    return pd.DataFrame(rows).T, len(d)
