"""5-year rent forecasts from the DFFH Rental Report history (quarterly, 2000 onwards).

Several simple methods are compared on back-tests: pretend it is an earlier quarter (the
"origin"), forecast 5 years ahead, and compare with what really happened. The method with
the lowest error is then used for the real forecast, and its back-test errors give the
uncertainty band.

Every method sees the same thing in the back-test and in the real forecast: the last
FIT_WINDOW quarters of each area's log rent. So the model that is tested is the model
that is used.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

H = 20                   # forecast horizon: 5 years of quarters
FIT_WINDOW = 40          # quarters of history each method is fitted on (10 years)
CHECK_FROM = "2010Q1"    # coverage and bond-count screens are judged on 2010Q1 onwards
MIN_COVERAGE = 0.95      # share of quarters with a published median
MIN_COUNT = 100          # median new bonds per quarter; thinner areas have noisy medians
ORIGINS = ("2012Q4", "2014Q4", "2016Q4", "2018Q4", "2020Q3")   # spread out, not all COVID

METRO_REGIONS = {
    "Inner Melbourne", "Inner Eastern Melbourne", "Southern Melbourne",
    "Outer Western Melbourne", "North Western Melbourne", "North Eastern Melbourne",
    "Outer Eastern Melbourne", "South Eastern Melbourne", "Mornington Peninsula",
}


def _series(panel, property_type="All properties", min_coverage=MIN_COVERAGE,
            min_count=MIN_COUNT, check_from=CHECK_FROM):
    """Wide table of log rents: one column per DFFH area, one row per quarter (2000Q1 on).

    An area is kept only if, from check_from onwards, it has a median in at least
    min_coverage of quarters AND a median of at least min_count new bonds per quarter.
    Gaps inside a series are interpolated; missing quarters at the start are left blank,
    so no area is given a made-up early history.
    """
    p = panel[panel["property_type"] == property_type]
    rent = p.pivot_table(index="quarter", columns="area", values="median_rent")
    count = p.pivot_table(index="quarter", columns="area", values="count")
    recent = rent.index >= pd.Period(check_from, "Q")
    coverage = rent[recent].notna().mean()
    thickness = count[recent].median()
    keep = coverage.index[(coverage >= min_coverage) & (thickness.reindex(coverage.index) >= min_count)]
    return np.log(rent[keep].interpolate(limit_area="inside"))


def screened_out(panel, property_type="All properties"):
    """Areas dropped by the screens in _series, and why (for the notebook)."""
    p = panel[panel["property_type"] == property_type]
    rent = p.pivot_table(index="quarter", columns="area", values="median_rent")
    count = p.pivot_table(index="quarter", columns="area", values="count")
    recent = rent.index >= pd.Period(CHECK_FROM, "Q")
    s = pd.DataFrame({"coverage": rent[recent].notna().mean(),
                      "median_bonds_per_qtr": count[recent].median()})
    s["dropped_for"] = np.select(
        [s["coverage"] < MIN_COVERAGE, s["median_bonds_per_qtr"] < MIN_COUNT],
        ["coverage", "too few bonds"], default="")
    return s[s["dropped_for"] != ""].sort_values("median_bonds_per_qtr")


def _arima(y, h):
    """ARIMA on log rent with drift; the order with the lowest AIC of three small ones."""
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in [(1, 1, 0), (2, 1, 0), (1, 1, 1)]:
            try:
                fit = ARIMA(y, order=order, trend="t").fit()
            except Exception:
                continue
            if best is None or fit.aic < best.aic:
                best = fit
    return best.forecast(h)


def _forecast(y, method, h=H):
    """Forecast h quarters of one log-rent series (numpy array, last FIT_WINDOW quarters)."""
    if method == "naive":            # baseline: rent stays where it is
        return np.repeat(y[-1], h)
    if method == "trend_10y":        # continue the average growth of the last 10 years
        g = (y[-1] - y[-41]) / 40
        return y[-1] + g * np.arange(1, h + 1)
    if method == "trend_5y":         # continue the average growth of the last 5 years
        g = (y[-1] - y[-21]) / 20
        return y[-1] + g * np.arange(1, h + 1)
    if method == "holt_damped":      # exponential smoothing, trend that slowly flattens
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y, trend="add", damped_trend=True).fit()
        return fit.forecast(h)
    if method == "arima":            # Lecture 7: ARIMA with drift
        return _arima(y, h)
    if method == "blend":            # average of the long-run trend and Holt
        return (_forecast(y, "trend_10y", h) + _forecast(y, "holt_damped", h)) / 2
    raise ValueError(method)


METHODS = ["naive", "trend_10y", "trend_5y", "holt_damped", "arima", "blend"]


def _window(col):
    """The last FIT_WINDOW+1 quarters of one area's log rent, or None if too short.
    (+1 because trend_10y needs the value 40 quarters back.)"""
    y = col.dropna().to_numpy()
    return y[-(FIT_WINDOW + 1):] if len(y) > FIT_WINDOW else None


def backtest(panel, origins=ORIGINS, methods=METHODS, property_type="All properties"):
    """One row per origin x area x method: 5-year-ahead forecast vs what happened."""
    logs = _series(panel, property_type)
    rows = []
    for origin in origins:
        o = pd.Period(origin, "Q")
        train = logs[logs.index <= o]
        target = o + H
        if target not in logs.index:
            continue
        for area in logs.columns:
            y = _window(train[area])
            actual_log = logs.at[target, area]
            if y is None or np.isnan(actual_log):
                continue
            actual = np.exp(actual_log)
            for m in methods:
                pred = np.exp(_forecast(y, m)[-1])
                rows.append({"origin": origin, "area": area, "method": m,
                             "actual_5y": actual, "forecast_5y": pred,
                             "pct_error": (pred / actual - 1) * 100,
                             "abs_pct_error": abs(pred / actual - 1) * 100})
    return pd.DataFrame(rows)


def score(bt):
    """Mean absolute % error by method, overall and per origin (lower is better)."""
    by_origin = bt.pivot_table(index="method", columns="origin", values="abs_pct_error")
    by_origin.insert(0, "overall", bt.groupby("method")["abs_pct_error"].mean())
    by_origin["bias"] = bt.groupby("method")["pct_error"].mean()   # + means over-forecast
    return by_origin.sort_values("overall").round(2)


def forecast_areas(panel, method, property_type="All properties"):
    """Quarterly history + 5-year forecast for every DFFH area that passes the screens."""
    logs = _series(panel, property_type)
    last = logs.index.max()
    future = pd.period_range(last + 1, periods=H, freq="Q")
    region = panel.drop_duplicates("area").set_index("area")["region"]
    rows = []
    for area in logs.columns:
        hist = logs[area].dropna()
        y = _window(hist)
        if y is None:
            continue
        f = np.exp(_forecast(y, method))
        rows.append(pd.DataFrame({"area": area, "quarter": hist.index,
                                  "rent": np.exp(hist.to_numpy()), "kind": "actual"}))
        rows.append(pd.DataFrame({"area": area, "quarter": future, "rent": f, "kind": "forecast"}))
    out = pd.concat(rows, ignore_index=True)
    out["region"] = out["area"].map(region)
    out["metro"] = out["region"].isin(METRO_REGIONS)
    return out


def summarise(long, bt, method, level=0.8):
    """One row per area: rent now, rent in 5 years, growth (% a year and $ a week), and an
    uncertainty band from the chosen method's back-test errors (default: middle 80%)."""
    now = long[long["kind"] == "actual"].groupby("area").last()
    end = long[long["kind"] == "forecast"].groupby("area").last()
    s = pd.DataFrame({"region": now["region"], "metro": now["metro"],
                      "rent_now": now["rent"], "quarter_now": now["quarter"],
                      "rent_5y": end["rent"], "quarter_5y": end["quarter"]})
    s["growth_pa_pct"] = ((s["rent_5y"] / s["rent_now"]) ** (1 / 5) - 1) * 100
    s["growth_5y_pct"] = (s["rent_5y"] / s["rent_now"] - 1) * 100
    s["change_per_week"] = s["rent_5y"] - s["rent_now"]
    # pct_error = forecast/actual - 1, so actual = forecast / (1 + error)
    e = bt.loc[bt["method"] == method, "pct_error"] / 100
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    s["rent_5y_low"] = s["rent_5y"] / (1 + e.quantile(hi_q))
    s["rent_5y_high"] = s["rent_5y"] / (1 + e.quantile(lo_q))
    s["growth_pa_low"] = ((s["rent_5y_low"] / s["rent_now"]) ** (1 / 5) - 1) * 100
    s["growth_pa_high"] = ((s["rent_5y_high"] / s["rent_now"]) ** (1 / 5) - 1) * 100
    return s.sort_values("growth_pa_pct", ascending=False).reset_index()


def what_if(long, summary, weeks_per_quarter=13):
    """'What if you had rented out one home in each area from the last actual quarter (2025Q3)?'

    One row per area: rent collected over the 5 forecast years (income_5y), and how much of that comes from rent
    growth, i.e. compared with rent staying at today's level (extra_from_growth). The _low/_high columns use the
    80% range from summarise(): the forecast path is scaled smoothly so it ends at rent_5y_low / rent_5y_high.
    Rent income only, before costs; not a return on investment (no purchase prices)."""
    f = long[long["kind"] == "forecast"].copy()
    f["k"] = f.groupby("area").cumcount() + 1                      # quarters ahead: 1..H
    s = summary.set_index("area")
    f = f.join(s[["rent_now", "rent_5y", "rent_5y_low", "rent_5y_high"]], on="area")
    out = pd.DataFrame(index=s.index)
    flat = s["rent_now"] * weeks_per_quarter * H                   # rent never changes
    for suffix, end in [("", "rent_5y"), ("_low", "rent_5y_low"), ("_high", "rent_5y_high")]:
        path = f["rent"] * (f[end] / f["rent_5y"]) ** (f["k"] / H)
        out["income_5y" + suffix] = (path * weeks_per_quarter).groupby(f["area"]).sum()
        out["extra_from_growth" + suffix] = out["income_5y" + suffix] - flat
    return out
