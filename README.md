# Satellite Classification for Wildfire Prediction and Management

Classifying wildfire intensity from NASA MODIS satellite data using a 
reproducible R targets pipeline with Random Forest and XGBoost models.

## Overview

In this project, I build a production-style machine learning pipeline to classify 
wildfire intensity from satellite-derived thermal and temporal features. The 
full pipeline is reproducible end-to-end using the `targets` package - see 
*Pipeline Architecture* below for setup and data download instructions.

**Fire Intensity classification** (Low / Medium / High) is the primary, 
fully-evaluated task in this project. Final models achieve **99.19% (RF) and 
99.09% (XGBoost)** accuracy on held-out Australian test data, and **98.11% 
(RF) and 98.15% (XGBoost)** accuracy when tested on a separate global dataset 
spanning 207 countries (see "Global Generalization Test" below for full 
results and a key finding about intensity threshold generalization).

A secondary **Day vs Night** fire-activity model (`model_rf_day`) is also 
trained as part of the pipeline but is not currently evaluated or reported 
on in this README.

## Datasets

**Australia & New Zealand (training data):**
[Fires From Space - Australia & New Zealand](https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland) 
— Kaggle dataset of MODIS satellite fire detections across Australia and New Zealand.

**Global 2024 (generalization test data):**
[NASA FIRMS Fire Archive Download](https://firms.modaps.eosdis.nasa.gov/data/download/DL_FIRE_M-C61_762558.zip)
- Data Source: MODIS C6.1
- Date Range: 2024-01-01 to 2024-12-31
- Output Format: CSV
- Area of Interest: -180,-90,180,90 (global; full latitude/longitude extent)
- Downloaded as 207 separate per-country CSV files, combined into a single file
  ~4.85 million row dataset (see *Pipeline Architecture* for combination steps).

## Methodology Notes & Design Decisions

### Intensity Classification Thresholds
`intensity_class` (Low/Medium/High) was originally derived from FRP quantiles 
computed on whatever dataset was passed in; meaning "High" meant "top third 
of FRP values in this specific dataset," not a fixed real-world severity level. 
This breaks down when testing on new data with a different FRP distribution. 
*Think of a small dataset from a less fire-intense region that would have its own, 
artificially inflated "High" class.

**Fix:** FRP cutoffs are now calculated once from the training dataset and 
passed as fixed arguments to `load_and_clean_fires()` for any subsequent 
dataset (test splits, new regions, future years).

*In progress: cross-validating these training-derived cutoffs against 
domain-standard FRP severity thresholds from fire science literature, to 
confirm they're physically reasonable rather than purely an artifact of 
this dataset's distribution.*

[Severe Fire Danger Index: A Forecastable Metric to Inform Firefighter and Community Wildfire
Risk Management](https://www.fs.usda.gov/rm/pubs_journals/2019/rmrs_2019_jolly_m001.pdf)

### Timezone Calculation
Local time was originally approximated geometrically (`longitude / 15` hours), 
which assumes time zones track longitude smoothly. This holds reasonably well 
for Australia (whose real time zones roughly follow its longitude bands) but 
breaks down for countries where legal time zones don't match longitude 
*All of China uses one time zone despite spanning ~60° of longitude.

**Fix:** Local time is now calculated via `lutz::tz_lookup_coords()`, which 
maps each lat/lon point to its real administrative timezone, combined with 
`lubridate::with_tz()` to get the correct local time (including 
daylight saving adjustments where applicable).

### Spatial Grouping Resolution
Two separate spatial groupings are used in the pipeline:
- **Fine-grained (`round(lat/lon, 2)`, ~1.1km):** "Fires in same place;" orders
detections at the same physical point through time, for `time_since_last` calculation.
- **Coarse-grained (`round(lat/lon, 1)`, ~11km, 2-hour bins):** "Fires in same area;" finds nearby 
  fire detections to compute neighbor-cluster statistics (`neighbor_mean_frp`, 
  `neighbor_count`).

**Known limitation:** longitude-based rounding represents a shrinking 
real-world distance at higher latitudes (longitude lines converge at the 
poles), while latitude-based rounding stays constant (~111km/degree) 
everywhere. This is independent of the timezone fix above. Not currently 
corrected for; acceptable for Australia's latitude range, but should be 
revisited before drawing conclusions from data at 60°+ latitude (relevant 
now that global MODIS data is being tested).

## Global Generalization Test (2024 MODIS Archive, 207 Countries)

To test whether the Australia-trained models generalize reliably beyond their 
original region, both models were evaluated against the full 2024 NASA FIRMS MODIS 
archive (~4.85 million fire detections spanning 207 countries and territories).

**Results:**
- Random Forest: 98.11% accuracy (Kappa: 0.9711)
- XGBoost: 98.15% accuracy (Kappa: 0.9718)

Compared to Australia-only test performance (99.19% RF / 99.09% XGBoost), 
both models held up well, dropping only about 1 percentage point in overall 
accuracy on entirely unseen, global data.

### Key Finding: Australia's Low/Medium Boundary Doesn't Fully Generalize

The small amount of error that does appear is not evenly spread across 
classes, but concentrates specifically at the **Low/Medium intensity boundary**:

| Metric | Australia | Global |
|---|---|---|
| "Low" Sensitivity | 99.67% | 96.83% (RF) / 96.87% (XGB) |
| "Medium" Pos. Pred. Value | ~99% | 94.80% (RF) / 94.95% (XGB) |
| "High" Balanced Accuracy | ~99.5% | 98.4% (RF) / 98.4% (XGB) |

"High" intensity classification stayed nearly as accurate globally as it was 
on Australia. The drop is concentrated in fires near the Low/Medium cutoff 
specifically.

**Why this happens:** this project deliberately calculates FRP cutoffs for 
Low/Medium/High *once*, from Australia's training data, and reuses those 
same fixed thresholds for any new dataset rather than recalculating a new 
"top third" / "bottom third" cutoff separately for each dataset tested. 
This was a deliberate design choice: recalculating thresholds per-dataset 
would have hidden this exact finding, since each new dataset's "Low" and "Medium" 
classes would have redefined themselves around that dataset.

Because the threshold stayed fixed, this test reveals something real: **fires 
worldwide cluster differently around the Low/Medium FRP boundary than 
Australian fires do.** Many fires elsewhere in the world sit closer to that 
specific boundary than is typical in Australia, making them genuinely harder 
to classify on one side or the other based on a boundary calibrated 
to Australian fire behavior. This is not a bug in the pipeline; it's a 
legitimate difference in how fire intensity is distributed in Australia versus globally, 
revealed by the fixed-threshold difference. It's also a meaningful limitation to keep 
in mind if these specific thresholds were ever applied to inform real 
decisions *outside Australia*.

**Model adjustment required for global testing:** the Random Forest model 
originally included `satellite` and `type` (fire type: vegetation, volcano, 
other, offshore) as predictors. Since Australia's training data is almost 
entirely `type == 0` (vegetation fires), the trained model had only ever seen 
one factor level for `type`. Global data includes other fire types, such as
monitored volcanic hotspots, causing the model to fail with a "new factor 
levels" error during prediction. Feature importance analysis confirmed `type` 
contributed trivial (in one case negative) predictive value on Australia's 
data; so it was removed as a predictor, aligning the RF model's feature set 
with XGBoost's. This had no meaningful effect on Australia test accuracy 
(99.19% after removal vs. 98.6% in the original model), resolving generalization 
failure and facilitated reliable performance on new data.

## Pipeline Architecture

Built with the [`targets`](https://books.ropensci.org/targets/) package for 
full pipeline reproducibility — every step from raw data to trained models 
to evaluation plots is defined as a tracked, cached target in `_targets.R`.

### Setup

1. Clone this repository.
2. Open `Satellite_Wildfire_Pipeline.Rproj` in RStudio (ensures working 
   directory and relative paths resolve correctly).
3. Install required packages:
```r
   install.packages(c(
     "tidyverse", "lubridate", "readxl", "readr", "caret", "lutz",
     "randomForest", "rpart", "rpart.plot", "pheatmap", "xgboost",
     "data.table", "targets"
   ))
```

### Getting the data

This pipeline expects two raw data files in `data/raw/`:

1. **`Fires_From_Space_Australia_Dataset.xlsx`** — download from the 
   [Kaggle Australia/NZ dataset](https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland) 
   linked in *Datasets* above, and place in `data/raw/`.

2. **`modis_2024_global.csv`** — download the 
   [NASA FIRMS 2024 global MODIS archive](https://firms.modaps.eosdis.nasa.gov/data/download/DL_FIRE_M-C61_762558.zip) 
   linked in *Datasets* above. This downloads as 207 separate per-country CSVs; 
   combine them into one file before placing in `data/raw/`:
```r
   library(dplyr)
   library(purrr)
   
   csv_files <- list.files("path/to/downloaded/modis/folder", 
                            pattern = "\\.csv$", full.names = TRUE)
   combined_modis <- map_dfr(csv_files, read.csv)
   write.csv(combined_modis, "data/raw/modis_2024_global.csv", row.names = FALSE)
```

### Running the pipeline

```r
source("R/functions.R")
library(targets)
tar_make()
```

`targets` will build everything in correct dependency order: data cleaning → 
train/test split → model training (Random Forest, XGBoost, Decision Tree, 
PCA-based RF) → evaluation on both the Australia test split and the global 
dataset → diagnostic plots

### Viewing results

```r
tar_read(global_rf_confusion)
tar_read(global_xgb_confusion)
tar_visnetwork()  # visual dependency graph of the full pipeline
```

**Note:** the combined global CSV is large (~400MB) and is excluded from 
version control via Git LFS / `.gitignore`. See repository setup notes if 
cloning and reproducing the global test from scratch.
