# SatelliteClassificationForWildfirePredictionAndManagement
Exploring and analyzing a Kaggle dataset using Data Mining methods (mainly classification and visualization)

# Satellite Classification of Wildfire Behavior

## Overview
This project explores whether satellite-derived data can be used to classify wildfire behavior. Specifically, it examines two questions:

- Can satellite data distinguish between day and night fire activity?
- Can it classify wildfire intensity (Low, Medium, High)?

The goal is to evaluate how much meaningful structure exists in satellite thermal and temporal features.

---

## Dataset
Source: https://www.kaggle.com/datasets/carlosparadis/fires-from-space-australia-and-new-zeland  

The dataset contains satellite observations of fires across Australia and New Zealand.

Key variables:
- FRP (Fire Radiative Power): measures fire intensity (MW)
- Brightness temperature (bright_t31): thermal signal from fires
- Timestamp (date and time of detection)
- Latitude and longitude

---

## Data Processing
- Converted timestamps into local solar time  
- Created temporal features:
  - Hour of day  
  - Diurnal cycle (sin transformation)  
  - Time since last detection  
- Filtered unrealistic values (brightness temperature outside physical range)  
- Created classification labels:
  - Day vs Night  
  - Fire intensity (Low, Medium, High based on FRP)

---

## Methods
Classification models were used to test whether satellite features can separate fire behavior categories:

- Decision Trees (interpretable thresholds)
- Random Forest (primary model)
- Logistic Regression (baseline comparison)
- PCA (visualization and dimensionality reduction)

Evaluation was performed using confusion matrices and class-level performance metrics.

---

## Results

### Day vs Night Classification
- Satellite features capture some day/night structure
- However, misclassification rates are high
- Conclusion: signal exists but is not strong enough for reliable classification

---

### Fire Intensity Classification
- Low intensity fires: classified most accurately
- Medium intensity fires: moderate performance with overlap
- High intensity fires: poorly detected due to rarity and overlap

Key insight:
- Brightness temperature and temporal features are the strongest predictors
- Fire intensity classes overlap significantly in feature space

---

## Key Findings
- Satellite data contains meaningful thermal and temporal signals
- Low and medium fires can be classified reasonably well
- High-intensity fires are rarely detected correctly
- Day vs night differences exist but are not cleanly separable

---

## Limitations
- High-intensity fires are rare → limited training examples
- Significant overlap between classes
- Satellite features alone may not fully capture fire dynamics

---

## Conclusion
Satellite-derived features provide useful information for understanding wildfire behavior. However, while common fire conditions are classified reasonably well, rare extreme events remain difficult to detect.

Future improvements in sensor resolution and data availability could significantly improve classification performance.

---

## How to Run
1. Download dataset from Kaggle
2. Update file path in script
3. Install required R packages:
   - tidyverse  
   - lubridate  
   - readxl  
   - caret  
   - randomForest  
   - rpart  
   - pheatmap  
4. Run analysis script

---

## Author
Charlotte Dickson
