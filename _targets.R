# _targets.R
library(targets)

tar_option_set(
  packages = c(
    "tidyverse", "lubridate", "readxl", "caret", 
    "randomForest", "rpart", "rpart.plot", "pheatmap", "xgboost"  # <-- add
  )
)

# Source the single file directly
source("R/functions.R")

list(
  # 1. FILE DEPENDENCY
  tar_target(
    wildfire_data_file,
    "/Users/litchar/Library/Mobile Documents/com~apple~Numbers/Documents/Fires_From_Space_Australia_Dataset.xlsx",
    format = "file"
  ),
  
  # 2. TRANSFORMATIONS
  tar_target(
    cleaned_fires,
    load_and_clean_fires(wildfire_data_file)
  ),
  
  tar_target(
    split_datasets,
    split_wildfire_data(cleaned_fires, p = 0.7)
  ),
  
  # 3. GRAPHICS
  tar_target(
    plot_eda_box,
    plot_diurnal_boxplots(cleaned_fires)
  ),
  
  tar_target(
    plot_eda_thermal,
    plot_thermal_signals(cleaned_fires)
  ),
  
  # 4. DIMENSIONAL REDUCTION
  tar_target(
    pca_results,
    compute_pca(split_datasets$train)
  ),
  
  # 5. ML TARGETS
  tar_target(
    model_rf_day,
    train_rf_day(split_datasets$train)
  ),
  
  tar_target(
    model_rf_intensity,
    train_rf_intensity(split_datasets$train)
  ),
  
  tar_target(
    model_xgb_intensity,
    train_xgb_intensity(split_datasets$train)
  ),
  
  tar_target(
    model_decision_tree,
    train_decision_tree(split_datasets$train)
  ),
  
  tar_target(
    model_rf_pca,
    train_rf_pca(pca_results, split_datasets$train)
  ),
  
  # 6. PIPELINE VISUAL PERFORMANCE ARTIFACTS
  tar_target(
    plot_ml_predictions,
    plot_intensity_predictions(model_rf_intensity, split_datasets$test)
  ),
  
  tar_target(
    plot_ml_boundaries,
    plot_pca_boundaries(pca_results, model_rf_pca, split_datasets$train)
  ),
  
  tar_target(
    plot_ml_heatmap,
    plot_pheatmap_cm(model_rf_intensity, split_datasets$test)
  ),
  
  # 7. DIAGNOSTICS
  tar_target(
    misclass_diagnostics,
    get_misclass_data(split_datasets$test, model_rf_intensity)
  ),
  
  tar_target(
    plot_misclass_density,
    plot_misclass_distributions(misclass_diagnostics)
  )
)