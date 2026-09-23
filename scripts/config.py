"""All file locations for the project, in one place.

Every notebook and script imports paths from here, so nobody has to edit
hardcoded paths. Paths are built from the repo's top folder, so they work
whether you run code from notebooks/, scripts/ or the top folder.

Where each raw file comes from is listed in README.md ("Data setup").
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW = ROOT / "data" / "raw"
CURATED = ROOT / "data" / "curated"
PLOTS = ROOT / "plots"
EXTERNAL = RAW / "external"          # every downloaded public dataset lives here

# --- Rental listings, one entry per snapshot (source + collection month) ------------
# To add the 2026 rent.com.au data, run the scraper's merge step; the file below
# then exists and every downstream notebook picks it up automatically.
DOMAIN_2025 = RAW / "domain" / "Data" / "vic_rentals_all.csv"
POSTCODES = RAW / "domain" / "Data" / "postcodes.csv"
RENTCOMAU_2026 = RAW / "rentcomau" / "2026-09" / "vic_rentals_rentcomau.csv"

# --- External datasets (all public) -------------------------------------------------
SA2_ZIP = EXTERNAL / "SA2_2021_AUST_SHP_GDA2020.zip"                      # ABS ASGS 2021
ABS_ERP = EXTERNAL / "32180DS0003_2001-25.xlsx"                             # ABS population 2001-25
ABS_INCOME = EXTERNAL / "abs_income_table1_2018-19_to_2022-23.xlsx"        # ABS personal income
VIF_SA2 = EXTERNAL / "VIF2023_SA2_Pop_Hhold_Dwelling_Projections_to_2036_Release_2.xlsx"
GTFS_ZIP = EXTERNAL / "gtfs.zip"                                            # PTV timetable
SCHOOLS = EXTERNAL / "dv402-SchoolLocations2025.csv"                        # Vic Dept of Education
OSM_DIR = EXTERNAL / "osm"                                                  # OpenStreetMap extracts
ROUTE_CACHE = CURATED / "route_cache.parquet"                               # saved car routes

# --- Curated outputs ------------------------------------------------------------------
LISTINGS = CURATED / "listings.parquet"                  # all snapshots, shared columns
LISTINGS_FEATURES = CURATED / "listings_features.parquet"  # + SA2 and accessibility
SUBURB_FEATURES = CURATED / "suburb_features.parquet"    # one row per suburb x snapshot
SA2_FEATURES = CURATED / "sa2_features_v2.parquet"
DFFH_PANEL = CURATED / "dffh_rent_panel.parquet"
DFFH_CROSSWALK = CURATED / "dffh_area_suburb_crosswalk.csv"
FORECAST = CURATED / "rent_forecast_5yr.parquet"

MELBOURNE_CBD = (-37.8136, 144.9631)   # lat, lon (Flinders St / Swanston St corner)

for folder in (CURATED, PLOTS, OSM_DIR):
    folder.mkdir(parents=True, exist_ok=True)
