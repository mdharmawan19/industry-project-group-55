"""Question 1 modelling: which internal and external features predict rent.

Used by notebooks/06_rent_drivers.ipynb. The notebook sets out the method; this module holds the
pieces it repeats: preparing the model table, building one pipeline per model, grouped
cross-validation, permutation importance and what-if dollar values.

Every model is a scikit-learn Pipeline, so anything learned from data (scaling, filling in missing
values, tuning) is fitted on training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

# ------------------------------------------------------------------------------------------
# Features and groups
# ------------------------------------------------------------------------------------------
LISTED = ["has_air_conditioning", "has_heating", "has_dishwasher", "has_ensuite", "has_floorboards",
          "has_broadband", "has_balcony_deck", "has_courtyard_garden", "has_outdoor_entertaining",
          "has_pool", "has_gym", "has_study", "has_rumpus_room", "has_shed", "has_furnished",
          "has_pets_allowed", "has_security", "has_secure_parking"]          # recorded by both websites

GROUPS = {
    "Size": ["bedrooms", "bathrooms", "carspaces"],
    "Property type": ["property_group"],
    "Comfort and appliances": ["has_air_conditioning", "has_heating", "has_dishwasher", "has_ensuite",
                               "has_floorboards", "has_broadband"],
    "Outdoor and extras": ["has_balcony_deck", "has_courtyard_garden", "has_outdoor_entertaining", "has_pool",
                           "has_gym", "has_study", "has_rumpus_room", "has_shed", "has_furnished",
                           "has_pets_allowed", "has_security", "has_secure_parking"],
    "City access": ["cbd_route_km", "is_metro"],
    "Transport": ["train_km", "bus_stop_km"],   # tram stop distance left out: it repeats CBD distance (rank corr. 0.96)
    "Local services": ["school_km", "hospital_km", "shop_km", "park_km"],
    "Dining and entertainment": ["entertainment_within_1km"],
    "Area socio-economics": ["MEDIAN_INCOME_2022_23", "INCOME_GROWTH_2018_19_2022_23", "POP_GROWTH_2021_2025",
                             "VIF_GROWTH_2026_2031"],
}
INTERNAL_GROUPS = ["Size", "Property type", "Comfort and appliances", "Outdoor and extras"]
SIDE = {g: ("internal" if g in INTERNAL_GROUPS else "external") for g in GROUPS}
CONTROLS = ["source_2026", "has_feature_list"]        # in every model, never ranked
FEATURES = [c for cols in GROUPS.values() for c in cols]

DISTANCES = ["cbd_route_km", "train_km", "bus_stop_km", "school_km", "hospital_km", "shop_km", "park_km",
             "entertainment_within_1km"]                     # skewed: log(1 + x) in linear models
AREA = ["MEDIAN_INCOME_2022_23", "INCOME_GROWTH_2018_19_2022_23", "POP_GROWTH_2021_2025", "VIF_GROWTH_2026_2031"]
TYPES = ["Apartment/Unit", "House", "Townhouse", "Studio", "Other residential"]

LABELS = {   # readable names for tables and charts
    "bedrooms": "Bedrooms", "bathrooms": "Bathrooms", "carspaces": "Car spaces", "property_group": "Property type",
    **{c: c.replace("has_", "").replace("_", " ").capitalize() for c in LISTED},
    "cbd_route_km": "Drive to CBD (km)", "is_metro": "In Greater Melbourne", "train_km": "Nearest train station (km)",
    "bus_stop_km": "Nearest bus stop (km)", "school_km": "Nearest school (km)", "hospital_km": "Nearest hospital (km)",
    "shop_km": "Nearest shops (km)", "park_km": "Nearest park (km)",
    "entertainment_within_1km": "Dining and entertainment within 1 km",
    "MEDIAN_INCOME_2022_23": "Area median income ($)", "INCOME_GROWTH_2018_19_2022_23": "Area income growth",
    "POP_GROWTH_2021_2025": "Population growth 2021-25", "VIF_GROWTH_2026_2031": "Projected population growth 2026-31",
    "source_2026": "2026 listing (rent.com.au)", "has_feature_list": "Has a feature list",
}


def prepare(listings):
    """Model table from listings_final: usable listings with an SA2, the target, the raw features
    and the columns used only for splitting and reporting.

    Only fixed, row-by-row steps happen here (caps, flags). Steps that learn from data are in the
    pipelines, so they are fitted on training rows only."""
    d = listings[listings["usable"] & listings["SA2_CODE21"].notna()].reset_index(drop=True)
    X = pd.DataFrame({
        "bedrooms": d["bedrooms"].clip(upper=5),                         # 5+ as one level
        "bathrooms": d["bathrooms"].clip(upper=4),
        "carspaces": d["carspaces"].clip(upper=4),
        "property_group": d["property_group"].astype(str),
        **{c: d[c].astype(float) for c in LISTED},                       # 1 / 0 / blank = unknown
        "is_metro": d["is_metro"].map({True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}),
        **{c: d[c].astype(float) for c in DISTANCES if c != "is_metro"},
        **{c: d[c].astype(float) for c in AREA},
        "source_2026": (d["snapshot"] == "2026-09").astype(float),
        "has_feature_list": (~d["features_missing"].astype(bool)).astype(float),
    })
    y = np.log(d["weekly_rent"].to_numpy())
    meta = d[["listing_id", "snapshot", "suburb", "SA2_CODE21", "weekly_rent", "flag_long_listing",
              "lat", "lon"]].copy()
    meta["metro"] = X["is_metro"].astype(bool).to_numpy()
    return X, y, meta


# ------------------------------------------------------------------------------------------
# Pipelines: one per model family. `features` lets a group be left out (drop-group check).
# ------------------------------------------------------------------------------------------
def _bedroom_type(df):
    """Bedroom category x property type, e.g. '2|Apartment/Unit' (the linear models' size-type term)."""
    beds = df.iloc[:, 0].round().clip(0, 5).astype("Int64").astype(str).replace("<NA>", "missing")
    return (beds + "|" + df.iloc[:, 1].astype(str)).to_frame()


def _metro_cbd(df):
    """log(1 + CBD drive) separately for metro and regional listings (the CBD effect differs)."""
    km, metro = np.log1p(df.iloc[:, 0].to_numpy()), df.iloc[:, 1].to_numpy()
    return np.column_stack([km * metro, km * (1 - metro)])


def _bedroom_type_name(transformer, names):
    return np.array(["bedrooms_x_type"])


def _metro_cbd_names(transformer, names):
    return np.array(["log_cbd_km_metro", "log_cbd_km_regional"])


def linear_preprocessor(features):
    f = set(features)
    parts = []
    if {"bedrooms", "property_group"} <= f:
        parts.append(("size_type", make_pipeline(FunctionTransformer(_bedroom_type, feature_names_out=_bedroom_type_name),
                                                 OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20,
                                                               drop="first")),
                      ["bedrooms", "property_group"]))
    elif "property_group" in f:
        parts.append(("type", OneHotEncoder(handle_unknown="ignore", drop="first"), ["property_group"]))
    elif "bedrooms" in f:
        parts.append(("beds", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), ["bedrooms"]))
    counts = [c for c in ["bathrooms", "carspaces"] if c in f]
    if counts:
        parts.append(("counts", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                              StandardScaler()), counts))
    listed = [c for c in LISTED if c in f]
    if listed:                                              # unknown -> 0; `has_feature_list` marks it
        parts.append(("listed", SimpleImputer(strategy="constant", fill_value=0), listed))
    if {"cbd_route_km", "is_metro"} <= f:
        parts.append(("metro_cbd", make_pipeline(FunctionTransformer(_metro_cbd, feature_names_out=_metro_cbd_names), StandardScaler()),
                      ["cbd_route_km", "is_metro"]))
    elif "cbd_route_km" in f:
        parts.append(("cbd", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), ["cbd_route_km"]))
    flags = [c for c in ["is_metro"] + CONTROLS if c in f]
    if flags:
        parts.append(("flags", "passthrough", flags))
    dist = [c for c in DISTANCES if c in f and c != "cbd_route_km"]
    if dist:
        parts.append(("dist", make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler()), dist))
    if "MEDIAN_INCOME_2022_23" in f:
        parts.append(("income", make_pipeline(SimpleImputer(strategy="median"), FunctionTransformer(np.log, feature_names_out="one-to-one"),
                                              StandardScaler()), ["MEDIAN_INCOME_2022_23"]))
    growth = [c for c in AREA[1:] if c in f]
    if growth:
        parts.append(("growth", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), growth))
    return ColumnTransformer(parts, verbose_feature_names_out=False, sparse_threshold=0)


def tree_preprocessor(features):
    """Trees need no scaling or logs; property type is one-hot encoded; blanks stay blank."""
    f = list(features)
    parts = []
    if "property_group" in f:
        parts.append(("type", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["property_group"]))
    rest = [c for c in f if c != "property_group"]
    parts.append(("rest", "passthrough", rest))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


class GroupMedianBaseline(RegressorMixin, BaseEstimator):
    """Median log rent for the same bedrooms x property type x metro/regional (what an agent would quote).
    Falls back to bedrooms x type, then to the overall median, for combinations not seen in training."""
    keys = ["bedrooms", "property_group", "is_metro"]

    def fit(self, X, y):
        d = X[self.keys].assign(y=y)
        self.full_ = d.groupby(self.keys)["y"].median()
        self.part_ = d.groupby(self.keys[:2])["y"].median()
        self.all_ = float(np.median(y))
        return self

    def predict(self, X):
        k = pd.MultiIndex.from_frame(X[self.keys])
        p = self.full_.reindex(k).to_numpy(dtype=float, copy=True)
        miss = np.isnan(p)
        p[miss] = self.part_.reindex(pd.MultiIndex.from_frame(X.loc[miss, self.keys[:2]])).to_numpy()
        return np.where(np.isnan(p), self.all_, p)


SEED = 0
MODELS = {   # name: (complexity rank, builder(features) -> estimator, tuning grid)
    "Baseline": (0, lambda f: GroupMedianBaseline(), None),
    "Linear regression": (1, lambda f: Pipeline([("prep", linear_preprocessor(f)), ("model", LinearRegression())]),
                          None),
    "Ridge": (1, lambda f: Pipeline([("prep", linear_preprocessor(f)), ("model", Ridge())]),
              {"model__alpha": [0.03, 0.3, 3, 30]}),
    "Lasso": (1, lambda f: Pipeline([("prep", linear_preprocessor(f)), ("model", Lasso(max_iter=5000, tol=1e-3))]),
              {"model__alpha": [0.00003, 0.0001, 0.0003, 0.001]}),
    "Decision tree": (2, lambda f: Pipeline([("prep", tree_preprocessor(f)),
                                             ("model", DecisionTreeRegressor(random_state=SEED))]),
                      {"model__max_depth": [6, 10, 14], "model__min_samples_leaf": [20, 50]}),
    "Random forest": (3, lambda f: Pipeline([("prep", tree_preprocessor(f)),
                                             ("model", RandomForestRegressor(n_estimators=200, max_samples=0.5, max_features=0.33,
                                                                             random_state=SEED, n_jobs=-1))]),
                      {"model__min_samples_leaf": [5, 20]}),
    "Gradient boosting": (3, lambda f: Pipeline([("prep", tree_preprocessor(f)),
                                                 ("model", HistGradientBoostingRegressor(max_iter=400, min_samples_leaf=20,
                                                                                         l2_regularization=1.0,
                                                                                         random_state=SEED))]),
                          {"model__learning_rate": [0.05, 0.1], "model__max_leaf_nodes": [15, 31]}),
}
FAMILY = {"Baseline": "reference", "Linear regression": "linear", "Ridge": "linear", "Lasso": "linear",
          "Decision tree": "single tree", "Random forest": "tree ensemble", "Gradient boosting": "tree ensemble"}


def build(name, features, params=None):
    est = MODELS[name][1](features)
    return est.set_params(**params) if params else est


def fit_tuned(name, X, y, groups, features, inner_splits=3):
    """Fit one model on training rows; its settings are tuned by grouped CV inside those rows only."""
    grid = MODELS[name][2]
    if not grid:
        return build(name, features).fit(X[features], y), {}
    search = GridSearchCV(build(name, features), grid, cv=GroupKFold(n_splits=inner_splits),
                          scoring="neg_mean_absolute_error", n_jobs=1)
    search.fit(X[features], y, groups=groups)
    return search.best_estimator_, search.best_params_


# ------------------------------------------------------------------------------------------
# Accuracy, importance and dollar values
# ------------------------------------------------------------------------------------------
def scores(y_log, pred_log):
    """MAE and RMSE in $/week, median error as % of the asking rent, and R-squared on the log scale the
    models are fitted on."""
    actual, pred = np.exp(y_log), np.exp(pred_log)
    return {"MAE ($/wk)": np.mean(np.abs(pred - actual)),
            "median error (%)": 100 * np.median(np.abs(pred - actual) / actual),
            "RMSE ($/wk)": np.sqrt(np.mean((pred - actual) ** 2)),
            "R2 (log rent)": 1 - np.sum((y_log - pred_log) ** 2) / np.sum((y_log - y_log.mean()) ** 2)}


def permutation_importance(model, X, y, columns_sets, n_repeats=3, seed=SEED):
    """Rise in MAE ($/week) when each set of columns is shuffled together, breaking its link with rent.
    A set's columns are shuffled with the same row order, so their relationship to each other is kept."""
    rng = np.random.default_rng(seed)
    base = scores(y, model.predict(X))["MAE ($/wk)"]
    out = {}
    for name, cols in columns_sets.items():
        rises = []
        for _ in range(n_repeats):
            Xp = X.copy()
            order = rng.permutation(len(X))
            Xp[cols] = X[cols].to_numpy()[order]
            rises.append(scores(y, model.predict(Xp))["MAE ($/wk)"] - base)
        out[name] = float(np.mean(rises))
    return pd.Series(out)


def what_if(model, X, a, b):
    """Average change in predicted rent ($/week) between two versions of the same listings, b minus a.
    `a` and `b` map a column to a value, or to a function applied to that column (e.g. one more bedroom)."""
    def apply(changes):
        Xc = X.copy()
        for col, v in changes.items():
            Xc[col] = v(Xc[col]) if callable(v) else v
        return Xc
    return float(np.mean(np.exp(model.predict(apply(b))) - np.exp(model.predict(apply(a)))))
