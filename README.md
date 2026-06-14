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

## Pipeline Architecture

Built with the `targets` package for fully reproducible execution.
