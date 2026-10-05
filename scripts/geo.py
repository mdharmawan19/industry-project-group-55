"""Location features: SA2 districts, nearby amenities and car-route distances.

Everything works on a table with `lat` and `lon` columns, so it runs the same way for
2025 and 2026 listings. Downloads and car routes are saved to disk, so a re-run only
fetches what is new (important for API limits).
"""
import io
import time
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

import config

EARTH_RADIUS_KM = 6371.0088
UA = {"User-Agent": "MAST30034-student-project (University of Melbourne coursework)"}


# ----------------------------------------------------------------------------- SA2
def load_sa2_vic():
    sa2 = gpd.read_file(config.SA2_ZIP)
    sa2 = sa2[(sa2["STE_CODE21"] == "2") & sa2.geometry.notna()]
    return sa2[["SA2_CODE21", "SA2_NAME21", "SA3_NAME21", "SA4_NAME21", "GCC_NAME21", "AREASQKM21", "geometry"]]


def attach_sa2(df, sa2):
    """Add the SA2 district each listing's point falls inside."""
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    joined = gpd.sjoin(pts.to_crs(sa2.crs), sa2[["SA2_CODE21", "SA2_NAME21", "geometry"]],
                       how="left", predicate="within")
    joined = joined[~joined.index.duplicated()]          # points exactly on a border
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


def sa2_features():
    """Population, Victoria in Future 2023 projection, income, and Greater Melbourne vs the rest
    of Victoria, for every Victorian SA2.

    The SA2 data is built once, by notebooks/04_abs_sa2_crosswalk.ipynb
    (data/curated/sa2_features.csv). This function only reads it, turns growth over a period
    into growth per year, and adds the Greater Melbourne label from the SA2 boundary file.
    """
    f = pd.read_csv(config.SA2_FEATURES, dtype={"SA2_CODE21": str})
    per_year = lambda total, years: (1 + total) ** (1 / years) - 1
    out = pd.DataFrame({
        "SA2_CODE21": f["SA2_CODE21"],
        "POP_2025": f["POP_2025"],
        "POP_GROWTH_2021_2025": per_year(f["POP_GROWTH_2021_2025"], 4),          # % a year
        "POP_PROJ_2031": f["VIF_POP_2031"],
        "POP_PROJ_GROWTH_2026_2031": per_year(f["VIF_GROWTH_2026_2031"], 5),     # % a year, VIF2023
        "EARNERS_2022_23": f["EARNERS_2022_23"],
        "MEDIAN_INCOME_2018_19": f["MEDIAN_INCOME_2018_19"],
        "MEDIAN_INCOME_2022_23": f["MEDIAN_INCOME_2022_23"],
        "MEAN_INCOME_2022_23": f["MEAN_INCOME_2022_23"],
        "INCOME_GROWTH_2019_2023": per_year(f["INCOME_GROWTH_2018_19_2022_23"], 4),
    }).replace([np.inf, -np.inf], np.nan)
    out["POP_PROJ_SOURCE"] = "VIF2023"
    gcc = gpd.read_file(config.SA2_ZIP, ignore_geometry=True)[["SA2_CODE21", "GCC_NAME21"]]
    out = out.merge(gcc, on="SA2_CODE21", how="left")
    out["is_metro"] = out["GCC_NAME21"].eq("Greater Melbourne")
    return out


# ------------------------------------------------------------------ points of interest
def train_stations():
    """Metro (folder 1) and V/Line (folder 2) train stations from the PTV GTFS feed."""
    frames = []
    with zipfile.ZipFile(config.GTFS_ZIP) as outer:
        for mode in ("1", "2"):
            with zipfile.ZipFile(io.BytesIO(outer.read(f"{mode}/google_transit.zip"))) as inner:
                stops = pd.read_csv(inner.open("stops.txt"))
            if "location_type" in stops:
                parents = stops[stops["location_type"] == 1]
                stops = parents if len(parents) else stops[stops["parent_station"].isna()]
            frames.append(stops[["stop_name", "stop_lat", "stop_lon"]])
    st = pd.concat(frames).rename(columns={"stop_name": "name", "stop_lat": "lat", "stop_lon": "lon"})
    st["name"] = st["name"].str.replace(r"\s*\(.*\)$", "", regex=True).str.replace(
        r" (Railway )?Station.*$", "", regex=True)
    # one point per station name (metro and V/Line list some stations twice)
    return st.groupby("name", as_index=False)[["lat", "lon"]].mean()


def schools():
    s = pd.read_csv(config.SCHOOLS, encoding="utf-8-sig")
    return s.rename(columns={"School_Name": "name", "Y": "lat", "X": "lon",
                             "School_Type": "type"})[["name", "type", "lat", "lon"]].dropna()


OSM_QUERIES = {   # OpenStreetMap tags for each amenity group (Victoria only)
    "parks": 'nwr["leisure"~"^(park|nature_reserve)$"]',
    "shopping": 'nwr["shop"~"^(mall|department_store|supermarket)$"]',
    "entertainment": 'nwr["amenity"~"^(cinema|theatre|nightclub|pub|bar|restaurant|cafe|arts_centre)$"]',
}


def osm_pois(group, refresh=False):
    """Download one amenity group from OpenStreetMap (Overpass API) and cache it as CSV."""
    path = config.OSM_DIR / f"osm_{group}.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path)
    query = (f'[out:json][timeout:300];area["ISO3166-2"="AU-VIC"]->.vic;'
             f'({OSM_QUERIES[group]}(area.vic););out center tags;')
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query},
                      headers=UA, timeout=400)
    r.raise_for_status()
    rows = []
    for el in r.json()["elements"]:
        lat = el.get("lat", el.get("center", {}).get("lat"))
        lon = el.get("lon", el.get("center", {}).get("lon"))
        tags = el.get("tags", {})
        kind = tags.get("leisure") or tags.get("shop") or tags.get("amenity")
        if lat is not None:
            rows.append({"name": tags.get("name"), "kind": kind, "lat": lat, "lon": lon})
    out = pd.DataFrame(rows).drop_duplicates(["lat", "lon"])
    out.to_csv(path, index=False)
    return out


def _tree(pois):
    return BallTree(np.radians(pois[["lat", "lon"]].to_numpy()), metric="haversine")


def nearest(df, pois, prefix, k=1):
    """Straight-line km to the nearest POI (and its name). k>1 also returns candidates."""
    dist, idx = _tree(pois).query(np.radians(df[["lat", "lon"]].to_numpy()), k=k)
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_km"] = dist[:, 0] * EARTH_RADIUS_KM
    out[f"nearest_{prefix}"] = pois["name"].to_numpy()[idx[:, 0]]
    if k > 1:
        out[f"{prefix}_candidates"] = list(idx)
    return out


def count_within(df, pois, radius_km, name):
    counts = _tree(pois).query_radius(np.radians(df[["lat", "lon"]].to_numpy()),
                                      r=radius_km / EARTH_RADIUS_KM, count_only=True)
    return pd.Series(counts, index=df.index, name=name)


# ---------------------------------------------------------------------- car routes
OSRM = "https://router.project-osrm.org/table/v1/driving/"


def _route_table(sources, destinations):
    """Car distance (km) matrix from OSRM. OpenRouteService's matrix endpoint returns the
    same thing and can be swapped in here if you have an ORS key."""
    pts = list(sources) + list(destinations)
    coords = ";".join(f"{lon:.5f},{lat:.5f}" for lat, lon in pts)
    src = ";".join(str(i) for i in range(len(sources)))
    dst = ";".join(str(len(sources) + j) for j in range(len(destinations)))
    for attempt in range(4):
        r = requests.get(f"{OSRM}{coords}", headers=UA, timeout=60,
                         params={"sources": src, "destinations": dst, "annotations": "distance"})
        if r.status_code == 200:
            time.sleep(1.1)                        # public server: at most 1 request a second
            d = np.array(r.json()["distances"], dtype=float)
            return d / 1000
        time.sleep(5 * (attempt + 1))
    r.raise_for_status()


def route_features(points, stations, max_coords=90):
    """For each point (a suburb centre): car km to the Melbourne CBD and to the nearest
    train station by road (checking the 3 closest stations in a straight line).
    Results are cached in data/curated/route_cache.parquet by rounded coordinates."""
    pts = points[["lat", "lon"]].round(4).drop_duplicates().reset_index(drop=True)
    cache = pd.read_parquet(config.ROUTE_CACHE) if config.ROUTE_CACHE.exists() else \
        pd.DataFrame(columns=["lat", "lon", "cbd_route_km", "train_route_km", "train_route_station"])
    todo = pts.merge(cache[["lat", "lon"]], how="left", indicator=True)
    todo = todo[todo["_merge"] == "left_only"].drop(columns="_merge").reset_index(drop=True)

    if len(todo):
        cand = nearest(todo, stations, "train", k=3)["train_candidates"]
        todo["cbd_route_km"] = np.nan
        todo["train_route_km"] = np.nan
        todo["train_route_station"] = None
        # CBD: many origins, one destination per request
        for start in range(0, len(todo), max_coords - 1):
            chunk = todo.iloc[start:start + max_coords - 1]
            d = _route_table(chunk[["lat", "lon"]].to_numpy(), [config.MELBOURNE_CBD])
            todo.loc[chunk.index, "cbd_route_km"] = d[:, 0]
        # Stations: group nearby points so each request shares a few candidate stations
        order = todo.assign(first=[c[0] for c in cand]).sort_values("first").index
        batch = []
        def flush(batch):
            dest_idx = sorted({j for i in batch for j in cand[i]})
            d = _route_table(todo.loc[batch, ["lat", "lon"]].to_numpy(),
                             stations.iloc[dest_idx][["lat", "lon"]].to_numpy())
            for row, i in enumerate(batch):
                cols = [dest_idx.index(j) for j in cand[i]]
                best = cols[int(np.nanargmin(d[row, cols]))] if np.isfinite(d[row, cols]).any() else None
                if best is not None:
                    todo.loc[i, "train_route_km"] = d[row, best]
                    todo.loc[i, "train_route_station"] = stations.iloc[dest_idx[best]]["name"]
        for i in order:
            dests = {j for b in batch + [i] for j in cand[b]}
            if batch and len(batch) + 1 + len(dests) > max_coords:
                flush(batch)
                batch = []
            batch.append(i)
        if batch:
            flush(batch)
        cache = pd.concat([cache, todo], ignore_index=True)
        cache.to_parquet(config.ROUTE_CACHE, index=False)

    key = points[["lat", "lon"]].round(4)
    return key.merge(cache, on=["lat", "lon"], how="left").set_index(points.index)
