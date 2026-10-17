# industry-project-group-55

MAST30034 Project 2: Victorian rental prices. Which internal and external features drive rent (Q1), which suburbs
will grow fastest (Q2), and which suburbs are most liveable and affordable (Q3). Q1 is answered in `06_rent_drivers`,
Q2 in `03_dffh_panel` and Q3 in `07_liveability`.

**Start with [`notebooks/00_summary.ipynb`](notebooks/00_summary.ipynb)**: the answers to all three questions, our
approach, the issues we ran into, and the assumptions and limitations. It reads only the committed summaries in
`data/curated/` and `plots/`, so it runs without any raw data.

## Run order

| # | Notebook | Reads | Writes |
|---|---|---|---|
| 02 | `02_scraping` | rent.com.au (each person scrapes a share with `scripts/scrape_rentcomau.py`) | `data/raw/rentcomau/2026-09/vic_rentals_rentcomau.csv` |
| 01 | `01_data_curation` | Domain 2025 CSV, rent.com.au 2026 CSV | `data/curated/listings.parquet` (one row per listing, both years, 62 columns) |
| 03 | `03_dffh_panel` (**Q2**) | DFFH "Moving annual median rent by suburb and town" workbook, `postcodes.csv`; `suburb_sa2_crosswalk.csv` and `sa2_features.csv` (from 04, already in the repo); `listings.parquet` (from 01, optional) | 5-year rent forecast to 2030Q3 for 145 DFFH areas, chosen by back-test: `dffh_area_suburb_crosswalk.csv` (read by 01, 04, 07), `rent_forecast_areas.csv` (read by 07), `forecast_backtest.csv`; plots `dffh_rent_and_growth.png`, `population_vs_rent_growth.png`, `forecast_top10.png`, `what_if_top10.png` |
| 04 | `04_abs_sa2_crosswalk` | `listings.parquet`, DFFH crosswalk, SA2 boundaries, ABS population and income, VIF2023 | `listing_sa2.parquet` (85 columns), `sa2_features.parquet`, `suburb_sa2_crosswalk.csv`, SA2 maps |
| 05 | `05_geospatial_routing` | `listing_sa2.parquet`, PTV GTFS, school locations, OpenStreetMap, OSRM routes | `listings_final.parquet` (111 columns), station driving-distance map |
| 06 | `06_rent_drivers` (**Q1**) | `listings_final.parquet` | seven-model comparison, feature importance and pricing guide: `q1_model_comparison.csv`, `q1_importance.csv`, `q1_pricing_guide.csv`; plots `q1_group_importance.png`, `q1_feature_importance.png` |
| 07 | `07_liveability` (**Q3**) | `listings_final.parquet`, `rent_forecast_areas.csv` and `dffh_area_suburb_crosswalk.csv` (from 03) | liveability and affordability by suburb and the investment shortlist: `suburb_liveability.csv`; plots `liveability_vs_affordability.png`, `liveability_vs_growth.png`, `investment_heatmap_rent_to_income.png` |
| 00 | `00_summary` (**start here**) | committed `data/curated/*.csv` and `plots/` | nothing; summary of the whole project |

02 is only needed to re-scrape; with the merged CSV in place, start at 01. 03 takes about 3–5 minutes (the ARIMA
back-test). 03 and 04 use each other's outputs (04 reads 03's area-to-suburb crosswalk; 03 reads 04's population
files for its population test). Both are committed in `data/curated/`, so either can be run first. 03 does not need
the listings: without `listings.parquet` it reuses the postcode choices saved in the committed crosswalk. 06 takes
about 12 minutes on its first run and about a minute after that (model results are cached in `data/curated/q1_cache/`).
Run 07 after 03 and 05.
Shared code is in `scripts/` (`config.py` holds every file path, `listings.py` the cleaning rules, `geo.py` the
location features, `dffh.py` the DFFH helpers, `forecast.py` the Q2 forecasting methods, `drivers.py` the Q2
population test, `rent_model.py` the Q1 models, `style.py` the shared chart style).

## Data setup

Raw data is **not** in this repository (the Domain data may not be redistributed, and the scraped data is not
shared publicly). Put the files here:

| File | Location |
|---|---|
| Domain 2025 course data (`vic_rentals_all.csv`, `postcodes.csv`) | `data/raw/domain/Data/` |
| rent.com.au 2026 (merged by 02) | `data/raw/rentcomau/2026-09/vic_rentals_rentcomau.csv` |
| PTV GTFS (`gtfs.zip`), school locations (`dv402-SchoolLocations2025.csv`), DFFH "Moving annual median rent by suburb and town" workbook (`.xlsx`, from dffh.vic.gov.au/publications/rental-report) | `data/raw/external/` |
| For notebooks 04 and 05: ABS SA2 2021 boundaries, `SA2_2021_AUST_GDA2020.shp` (+ `.dbf`, `.prj`, `.shx`, `.xml`) | `data/raw/sa2_boundaries/` |
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
