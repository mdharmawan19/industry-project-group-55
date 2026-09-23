"""5-year rent forecasts from the DFFH Rental Report history (quarterly, 2000 onwards).

Several simple methods are compared on back-tests (pretend it is 2015 or 2020, forecast
5 years ahead, compare with what really happened). The method with the lowest error is
then refitted on all data and used for the real forecast.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

H = 20   # 5 years of quarters


def _series(panel, property_type="All properties", min_coverage=0.95, start="2010Q1"):
    """Wide table: one column per DFFH area, one row per quarter (log rents)."""
    p = panel[(panel["property_type"] == property_type)]
    wide = p.pivot_table(index="quarter", columns="area", values="median_rent")
    wide = wide[wide.index >= pd.Period(start, "Q")]
    wide = wide.loc[:, wide.notna().mean() >= min_coverage]
    return np.log(wide.interpolate(limit_direction="both"))


def _forecast(y, method, h=H):
    """Forecast h quarters of one log-rent series with the chosen method."""
    if method == "trend_10y":        # continue the average growth of the last 10 years
        g = (y.iloc[-1] - y.iloc[-41]) / 40
        return y.iloc[-1] + g * np.arange(1, h + 1)
    if method == "trend_5y":         # continue the average growth of the last 5 years
        g = (y.iloc[-1] - y.iloc[-21]) / 20
        return y.iloc[-1] + g * np.arange(1, h + 1)
    if method == "holt_damped":      # exponential smoothing, trend that slowly flattens
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(y.to_numpy(), trend="add", damped_trend=True).fit()
        return fit.forecast(h)
    if method == "blend":            # average of the long-run trend and Holt
        return (_forecast(y, "trend_10y", h) + _forecast(y, "holt_damped", h)) / 2
    raise ValueError(method)


METHODS = ["trend_10y", "trend_5y", "holt_damped", "blend"]


def backtest(panel, origins=("2015Q3", "2020Q3"), property_type="All properties"):
    """Error of each method when forecasting 5 years ahead from past start points."""
    logs = _series(panel, property_type, start="2000Q1")
    rows = []
    for origin in origins:
        o = pd.Period(origin, "Q")
        train = logs[logs.index <= o]
        test = logs[(logs.index > o) & (logs.index <= o + H)]
        if len(test) < H:
            continue
        for area in logs.columns:
            y = train[area].dropna()
            if len(y) < 41:
                continue
            actual = np.exp(test[area].iloc[-1])
            for m in METHODS:
                pred = np.exp(_forecast(y, m)[-1])
                rows.append({"origin": origin, "area": area, "method": m,
                             "actual_5y": actual, "forecast_5y": pred,
                             "abs_pct_error": abs(pred / actual - 1) * 100,
                             "pct_error": (pred / actual - 1) * 100})
    return pd.DataFrame(rows)


def forecast_areas(panel, method, property_type="All properties"):
    """Quarterly history + 5-year forecast for every DFFH area (long format)."""
    logs = _series(panel, property_type)
    last = logs.index.max()
    future = pd.period_range(last + 1, periods=H, freq="Q")
    rows = []
    for area in logs.columns:
        y = logs[area].dropna()
        f = np.exp(_forecast(y, method))
        rows.append(pd.DataFrame({"area": area, "quarter": y.index, "rent": np.exp(y.to_numpy()),
                                  "kind": "actual"}))
        rows.append(pd.DataFrame({"area": area, "quarter": future, "rent": f, "kind": "forecast"}))
    return pd.concat(rows, ignore_index=True)


def summarise(long, band_pct):
    """One row per area: rent now, rent in 5 years, yearly growth, and an error band
    based on the back-test error (e.g. +/- 12%)."""
    now = long[long["kind"] == "actual"].groupby("area").last()
    end = long[long["kind"] == "forecast"].groupby("area").last()
    s = pd.DataFrame({"rent_now": now["rent"], "quarter_now": now["quarter"],
                      "rent_5y": end["rent"], "quarter_5y": end["quarter"]})
    s["growth_5y_pct"] = (s["rent_5y"] / s["rent_now"] - 1) * 100
    s["growth_pa_pct"] = ((s["rent_5y"] / s["rent_now"]) ** (1 / 5) - 1) * 100
    s["rent_5y_low"] = s["rent_5y"] * (1 - band_pct / 100)
    s["rent_5y_high"] = s["rent_5y"] * (1 + band_pct / 100)
    return s.sort_values("growth_pa_pct", ascending=False).reset_index()
