# industry-project-group-55

MAST30034 Project 2: Victorian rental prices. Which internal and external features drive rent (Q1), which suburbs
will grow fastest (Q2), and which suburbs are most liveable and affordable (Q3). Q1 is answered in `06_rent_drivers`
and Q2 in `03_dffh_panel`.

## Run order

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 02 | `02_scraping` | rent.com.au (each person scrapes a share with `scripts/scrape_rentcomau.py`) | `data/raw/rentcomau/2026-09/vic_rentals_rentcomau.csv` |
| 01 | `01_data_curation` | Domain 2025 CSV, rent.com.au 2026 CSV | `data/curated/listings.parquet` (one row per listing, both years, 62 columns) |
| 02b | `02b_geo_visualisation` | `listings.parquet` | Sprint 1 maps in `plots/` |
| 03 | `03_dffh_panel` (**Q2**) | DFFH "Moving annual median rent by suburb and town" workbook, `listings.parquet` | 5-year rent forecast to 2030Q3, chosen by back-test: `dffh_rent_panel.parquet`, `dffh_area_suburb_crosswalk.csv`, `dffh_yoy_growth.parquet`, `forecast_backtest.csv`, `rent_forecast_areas.csv`, `plots/forecast_top10.png`, `plots/dffh_rent_and_growth.png` |
| 04 | `04_abs_sa2_crosswalk` | `listings.parquet`, DFFH crosswalk, SA2 boundaries, ABS population and income, VIF2023 | `listing_sa2.parquet` (85 columns), `sa2_features.parquet`, `suburb_sa2_crosswalk.csv`, SA2 maps |
| 05 | `05_geospatial_routing` | `listing_sa2.parquet`, PTV GTFS, school locations, OpenStreetMap, OSRM routes | `listings_final.parquet` (111 columns), station driving-distance map |
| 06 | `06_rent_drivers` | `listings_final.parquet` | internal and external rent-driver rankings and charts |

02 is only needed to re-scrape; with the merged CSV in place, start at 01. 03 runs after 01 and takes about
3–5 minutes (the ARIMA back-test). Shared code is in `scripts/` (`config.py` holds every file path, `listings.py`
the cleaning rules, `geo.py` the location features, `dffh.py` the DFFH helpers, `forecast.py` the Q2 forecasting
methods).

## Data setup

Raw data is **not** in this repository (the Domain data may not be redistributed, and the scraped data is not
shared publicly). Put the files here:

| File | Location |
|---|---|
| Domain 2025 course data (`vic_rentals_all.csv`, `postcodes.csv`) | `data/raw/domain/Data/` |
| rent.com.au 2026 (merged by 02) | `data/raw/rentcomau/2026-09/vic_rentals_rentcomau.csv` |
| ABS SA2 2021 boundaries (`SA2_2021_AUST_SHP_GDA2020.zip`), PTV GTFS (`gtfs.zip`), school locations (`dv402-SchoolLocations2025.csv`), DFFH "Moving annual median rent by suburb and town" workbook (`.xlsx`, from dffh.vic.gov.au/publications/rental-report) | `data/raw/external/` |
| For notebook 04: `SA2_2021_AUST_GDA2020.shp` (+ `.dbf`, `.prj`, `.shx`, `.xml`) | `data/raw/sa2_boundaries/` |
| For notebook 04: `32180DS0003_2001-25.xlsx` (ABS population) | `data/raw/abs_population/` |
| For notebook 04: `Table 1 - Total income, earners and summary statistics by geography, 2018-19 to 2022-23.xlsx` | `data/raw/abs_income/` |
| For notebook 04: `VIF2023_SA2_Pop_Hhold_Dwelling_Projections_to_2036_Release_2.xlsx` | `data/raw/vic_population_projections/` |
| For notebook 04: `postcodes.csv` | `data/raw/` |

OpenStreetMap places (parks, shops, entertainment, hospitals) are downloaded by 05 on its first run and saved in
`data/raw/external/osm/`; car routes are saved in `data/curated/route_cache.parquet`.

Install packages with `pip install -r requirements.txt`.

## Rules for this public repository

- Never commit raw data, scraped data or listing-level files (`listings*.parquet`, `listing_sa2.parquet`). Only
  aggregated results and plots.
- API keys (e.g. `ors_config.env`) stay out of git.
