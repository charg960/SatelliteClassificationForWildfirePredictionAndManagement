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

To test whether the Australia-trained models generalize beyond their original 
region, both models were evaluated against the full 2024 NASA FIRMS MODIS 
archive — approximately 4.85 million fire detections spanning 207 countries 
and territories.

**Results:**
- Random Forest: 98.11% accuracy (Kappa: 0.9711)
- XGBoost: 98.15% accuracy (Kappa: 0.9718)

Compared to Australia-only test performance (99.19% RF / 99.09% XGBoost), 
both models held up well, dropping only about 1 percentage point in overall 
accuracy on entirely unseen, global data.

### Key Finding: Australia's Low/Medium Boundary Doesn't Fully Generalize

The small amount of error that does appear is not evenly spread across 
classes — it concentrates specifically at the **Low/Medium intensity boundary**:

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
same fixed numeric thresholds for any new dataset (see Methodology Notes 
above) — rather than recalculating a new "top third" / "bottom third" cutoff 
separately for each dataset tested. This was a deliberate design choice: 
recalculating thresholds per-dataset would have hidden this exact finding, 
since each new dataset's "Low" and "Medium" classes would have silently 
redefined themselves around whatever data happened to be in them.

Because the threshold stayed fixed, this test reveals something real: **fires 
worldwide cluster differently around the Low/Medium FRP boundary than 
Australian fires do.** Many fires elsewhere in the world sit closer to that 
specific boundary than is typical in Australia, making them genuinely harder 
to classify on one side or the other of that line using a boundary calibrated 
to Australian fire behavior. This is not a bug in the pipeline — it's a 
legitimate, fixed-threshold-revealed difference in how fire intensity is 
distributed in Australia versus globally, and a meaningful limitation to keep 
in mind if these specific thresholds were ever applied to inform real 
decisions outside Australia.

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

Built with the `targets` package for reproducible execution.
