# industry-project-group-55

MAST30034 Project 2: Victorian rental prices. We answer three questions for a real estate client:

| Question | Answered in |
|---|---|
| Q1. What are the most important internal and external features in predicting rental prices? | `06_rent_drivers` |
| Q2. What are the top 10 suburbs with the highest predicted rent growth? | `03_dffh_panel` |
| Q3. What are the most liveable and affordable suburbs? | `07_liveability` |

**Start with [`notebooks/00_summary.ipynb`](notebooks/00_summary.ipynb).** It gives the answers to all three
questions, our approach, the issues we ran into, and the assumptions and limitations. It reads only the summary tables
and charts committed to this repository, so it runs in seconds without any raw data.

## Quick start (no raw data needed)

Use **Python 3.10–3.13** (we used 3.10). Python 3.14 will not work with the pinned versions: numpy 2.2.6 and
scipy 1.15.3 have no 3.14 builds.

```bash
git clone https://github.com/mdharmawan19/industry-project-group-55.git
cd industry-project-group-55
python3.10 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab                         # then open notebooks/00_summary.ipynb and run all cells
```

Open notebooks from the `notebooks/` folder (Jupyter and VS Code do this by default): each notebook finds the shared
code in `scripts/` relative to that folder.

## Repository layout

```
notebooks/        00_summary (start here) and the analysis notebooks 01–07
scripts/          shared code used by the notebooks (see below)
data/raw/         raw data: not in git, see "Reproducing the full analysis"
data/curated/     outputs; only the summary tables listed below are in git
plots/            charts written by the notebooks
requirements.txt  pinned package versions
```

| Script | What it holds | Used by |
|---|---|---|
| `config.py` | every file path, in one place | every notebook except 04 (which sets its own paths) |
| `listings.py` | the cleaning rules for both websites | 01, 06 |
| `scrape_rentcomau.py` | the rent.com.au scraper (run in the terminal) | 02 |
| `geo.py` | distances, OpenStreetMap downloads and road routing | 05 |
| `dffh.py`, `forecast.py`, `drivers.py` | DFFH helpers, the Q2 forecasting methods, the Q2 population test | 03 |
| `rent_model.py` | the Q1 models, importance and dollar values | 06 |
| `style.py` | the shared chart style | 03, 05, 06, 07 |

## Reproducing the full analysis

### 1. Get the raw data

Raw data is **not** in this repository: the Domain data may not be redistributed, and our scraped data is not shared
publicly. Create these folders inside `data/raw/` and put each file there.

**Listings**
- **Domain, September 2025** (`vic_rentals_all.csv`, `postcodes.csv`): from the MAST30034 course data on Canvas.
  Put both in `data/raw/domain/Data/`, and a copy of `postcodes.csv` in `data/raw/` (used by 04).
- **rent.com.au, September 2026** (`vic_rentals_rentcomau.csv`): made by notebook 02 (our scrape), saved in
  `data/raw/rentcomau/2026-09/`.

**Rent history (Q2)**
- **DFFH "Moving annual rent by suburb", September quarter 2025** (`.xlsx`): from the
  [DFFH Rental Report](https://www.dffh.vic.gov.au/publications/rental-report). Put it in `data/raw/external/`.

**Areas and population (04, and the SA2 boundaries also for 05 and 06)**
- **SA2 2021 boundaries**, shapefile in GDA2020 (`SA2_2021_AUST_GDA2020.shp` with its `.dbf`, `.prj`, `.shx`), from the
  [ABS digital boundary files](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/access-and-downloads/digital-boundary-files).
  Unzip into `data/raw/sa2_boundaries/`.
- **Population by SA2** (`32180DS0003_2001-25.xlsx`), from
  [ABS Regional population](https://www.abs.gov.au/statistics/people/population/regional-population).
  Put it in `data/raw/abs_population/`.
- **Personal income by SA2** (`Table 1 - Total income, earners and summary statistics by geography, 2018-19 to 2022-23.xlsx`),
  from [ABS Personal Income in Australia 2022–23](https://www.abs.gov.au/statistics/labour/earnings-and-working-conditions/personal-income-australia/2022-23).
  Put it in `data/raw/abs_income/`.
- **Population projections** (`VIF2023_SA2_Pop_Hhold_Dwelling_Projections_to_2036_Release_2.xlsx`), from
  [Victoria in Future 2023](https://www.planning.vic.gov.au/land-use-and-population-research/victoria-in-future).
  Put it in `data/raw/vic_population_projections/`.

**Transport and schools (05)**
- **PTV timetable** (`gtfs.zip`), from [Transport Victoria GTFS Schedule](https://opendata.transport.vic.gov.au/dataset/gtfs-schedule).
  Put it in `data/raw/external/`.
- **School locations** (`dv402-SchoolLocations2025.csv`), from
  [DataVic School Locations 2025](https://discover.data.vic.gov.au/dataset/school-locations-2025).
  Put it in `data/raw/external/`.

OpenStreetMap places (parks, shops, entertainment, hospitals) and driving distances are downloaded by notebook 05 on
its first run (internet needed, **no API keys**). They are saved in `data/raw/external/osm/` and
`data/curated/route_cache.parquet`, so later runs reuse them.

### 2. Run the notebooks in this order

Every output goes to `data/curated/` (tables) or `plots/` (charts).

1. **`02_scraping`** *(optional)*: only needed to re-scrape rent.com.au. Each person runs
   `scripts/scrape_rentcomau.py` for their share of suburbs (needs internet); the notebook then merges everyone's
   files into `vic_rentals_rentcomau.csv`.
2. **`01_data_curation`**: cleans and stacks the 2025 and 2026 listings into `listings.parquet` (one row per
   listing, 62 columns). Without the scraped file, the table has 2025 only.
3. **`04_abs_sa2_crosswalk`**: places each listing in its SA2 and adds population, income and projections.
   Writes `listing_sa2.parquet` (85 columns), `sa2_features.parquet`, `suburb_sa2_crosswalk.csv` and the SA2 maps.
4. **`03_dffh_panel` (Q2)**: forecasts rent to 2030 for 145 DFFH areas. Writes `rent_forecast_areas.csv`,
   `forecast_backtest.csv`, `dffh_area_suburb_crosswalk.csv` and four charts. Takes about 3–5 minutes. It uses the
   committed `sa2_features.csv` if 04 has not been run.
5. **`05_geospatial_routing`**: adds distances to transport and services, and driving distances. Writes
   `listings_final.parquet` (111 columns). Needs internet on the first run.
6. **`06_rent_drivers` (Q1)**: compares seven models and ranks the features. Writes the three `q1_*.csv` tables and
   two charts. Takes about 15–20 minutes the first time and under a minute after (results are saved in
   `data/curated/q1_cache/`).
7. **`07_liveability` (Q3)**: scores liveability and affordability by suburb and ranks the investment shortlist.
   Writes `suburb_liveability.csv` and three charts. Run it after 03 and 05.
8. **`00_summary`**: the summary of the whole project. Reads only the committed tables and charts.

## What is in git, and what is not

**In git (summary tables only, safe to share):**

| File in `data/curated/` | Made by | Used by |
|---|---|---|
| `q1_model_comparison.csv`, `q1_importance.csv`, `q1_pricing_guide.csv` | 06 | 00 |
| `rent_forecast_areas.csv`, `forecast_backtest.csv` | 03 | 07, 00 |
| `dffh_area_suburb_crosswalk.csv` | 03 | 04, 07, 00 |
| `suburb_sa2_crosswalk.csv` | 04 | 03 |
| `sa2_features.csv` | an earlier run of 04 (04 now writes `sa2_features.parquet`) | 03, so it runs before 04 |
| `suburb_liveability.csv` | 07 | 00 |

The charts in `plots/` are written by the notebooks above, except `presentation_growth_map.png` and
`presentation_liveability_growth.png`, which were made for the presentation slides.

**Not in git:** raw and scraped data, listing-level files (`listings*.parquet`, `listing_sa2*`), and caches
(`q1_cache/`, `route_cache.parquet`, OpenStreetMap downloads). The `.gitignore` enforces this.

## Privacy, security and responsible scraping

- **No listing-level data in this public repository.** Only suburb- and area-level summaries and charts are
  committed, and notebooks show summaries rather than individual listings.
- **No API keys are needed.** Road distances use the public OSRM server. Any key file (e.g. `ors_config.env`) is
  listed in `.gitignore`.
- **The scraper is polite.** It follows robots.txt, waits about 5 seconds between requests (never less than 3), and
  identifies itself as a student project with a contact email.
