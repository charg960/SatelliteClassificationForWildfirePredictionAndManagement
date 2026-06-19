# Satellite Classification for Wildfire Prediction and Management

Classifying wildfire intensity from NASA MODIS satellite data using a 
reproducible R targets pipeline with Random Forest and XGBoost models.

## Overview

This project builds a production-style machine learning pipeline to classify 
wildfire intensity from satellite-derived thermal and temporal features. 
It uses the `targets` package for full pipeline reproducibility.

Two classification tasks are addressed:
- **Day vs Night** fire activity detection
- **Fire Intensity** classification (Low / Medium / High)

Final models achieve **98.6% (RF) and 99.4% (XGBoost)** accuracy on held-out test data.

Still in progress: Testing on other datasets.

## Dataset

**Source:** [Fires From Space - Australia & New Zealand](https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland)

MODIS satellite fire detections across Australia and New Zealand.

Key variables used:
- `frp` — Fire Radiative Power (MW), measure of fire intensity
- `bright_t31` — Thermal channel brightness temperature
- `scan * track` — Pixel footprint area (corrects for edge-of-swath distortion)
- `confidence` — NASA detection confidence score
- `satellite` — Terra vs Aqua sensor
- `type` — Fire type (vegetation, volcano, other)
- Timestamp, latitude, longitude

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
dataset (test splits, new regions, future years). This is cross-validated 
against domain-standard FRP severity thresholds from fire science literature
(source below) to confirm the training-derived cutoffs are 
physically reasonable, rather than purely an artifact of this dataset.
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
- **Coarse-grained (`round(lat/lon, 1)`, ~11km, 2-hour bins):** "Fires in same area;"finds nearby 
  fire detections to compute neighbor-cluster statistics (`neighbor_mean_frp`, 
  `neighbor_count`).

**Known limitation:** longitude-based rounding represents a shrinking 
real-world distance at higher latitudes (longitude lines converge at the 
poles), while latitude-based rounding stays constant (~111km/degree) 
everywhere. This is independent of the timezone fix above. Not currently 
corrected for; acceptable for Australia's latitude range, but should be 
revisited before drawing conclusions from data at 60°+ latitude (relevant 
now that global MODIS data is being tested).

## Pipeline Architecture

Built with the `targets` package for reproducible execution.
