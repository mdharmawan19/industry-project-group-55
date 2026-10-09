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
SA2_SHP = RAW / "sa2_boundaries" / "SA2_2021_AUST_GDA2020.shp"            # ABS ASGS 2021 (same copy as 04)
ABS_ERP = EXTERNAL / "32180DS0003_2001-25.xlsx"                             # ABS population 2001-25
ABS_INCOME = EXTERNAL / "abs_income_table1_2018-19_to_2022-23.xlsx"        # ABS personal income
VIF_SA2 = EXTERNAL / "VIF2023_SA2_Pop_Hhold_Dwelling_Projections_to_2036_Release_2.xlsx"
GTFS_ZIP = EXTERNAL / "gtfs.zip"                                            # PTV timetable
SCHOOLS = EXTERNAL / "dv402-SchoolLocations2025.csv"                        # Vic Dept of Education
OSM_DIR = EXTERNAL / "osm"                                                  # OpenStreetMap extracts
ROUTE_CACHE = CURATED / "route_cache.parquet"                               # saved car routes
# DFFH Rental Report "Moving annual median rent by suburb and town" workbook
# (dffh.vic.gov.au/publications/rental-report). Matched by name so a newer quarter's download works
# without code changes; the most recently downloaded one is used.
DFFH_SUBURB_XLSX = max(EXTERNAL.glob("Moving annual*rent*suburb*.xlsx"), key=lambda p: p.stat().st_mtime, default=None)

# --- Curated outputs ------------------------------------------------------------------
LISTINGS = CURATED / "listings.parquet"                  # all snapshots, shared columns
LISTING_SA2 = CURATED / "listing_sa2.parquet"          # built by 04: listings + SA2 + population/income/VIF2023
LISTINGS_FINAL = CURATED / "listings_final.parquet"    # built by 05_geospatial_routing: + location features
DFFH_PANEL = CURATED / "dffh_rent_panel.parquet"
DFFH_CROSSWALK = CURATED / "dffh_area_suburb_crosswalk.csv"
DFFH_YOY = CURATED / "dffh_yoy_growth.parquet"

MELBOURNE_CBD = (-37.8136, 144.9631)   # lat, lon (Flinders St / Swanston St corner)

for folder in (CURATED, PLOTS, OSM_DIR):
    folder.mkdir(parents=True, exist_ok=True)
