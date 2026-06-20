# _targets.R
library(targets)

tar_option_set(
  packages = c(
    "tidyverse", "lubridate", "readxl", "caret", "lutz", "data.table",
    "randomForest", "rpart", "rpart.plot", "pheatmap", "xgboost"
  )
)
source("R/functions.R")

list(
  # 1. FILE DEPENDENCIES — both raw data sources, grouped together
  tar_target(
    wildfire_data_file,
    "data/raw/Fires_From_Space_Australia_Dataset.xlsx",
    format = "file"
  ),
  
  tar_target(
    global_modis_file,
    "data/raw/modis_2024_global.csv",
    format = "file"
  ),
  
  # 2. CLEANING / TRANSFORMATIONS — Australia first (it generates the cutoffs),
  #    then global (which CONSUMES those cutoffs, so it must come after)
  tar_target(
    cleaned_fires,
    load_and_clean_fires(wildfire_data_file)
  ),
  
  tar_target(
    cleaned_global_fires,
    load_and_clean_fires(
      global_modis_file,
      frp_low_cutoff  = quantile(cleaned_fires$frp, 0.33, na.rm = TRUE),
      frp_high_cutoff = quantile(cleaned_fires$frp, 0.67, na.rm = TRUE)
    )
  ),
  
  tar_target(
    split_datasets,
    split_wildfire_data(cleaned_fires, p = 0.7)
  ),
  
  # 3. EXPLORATORY GRAPHICS — describe the (Australia) training data
  tar_target(
    plot_eda_box,
    plot_diurnal_boxplots(cleaned_fires)
  ),
  
  tar_target(
    plot_eda_thermal,
    plot_thermal_signals(cleaned_fires)
  ),
  
  # 4. DIMENSIONALITY REDUCTION
  tar_target(
    pca_results,
    compute_pca(split_datasets$train)
  ),
  
  # 5. MODEL TRAINING — all models trained on Australia split_datasets$train
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
  
  # 6. EVALUATION ON AUSTRALIA TEST SPLIT — same-distribution performance
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
  
  # 7. EVALUATION ON GLOBAL DATA — generalization / distribution-shift test
  #    (depends on models from section 5 AND cleaned_global_fires from section 2)
  tar_target(
    global_rf_predictions,
    predict(model_rf_intensity, cleaned_global_fires)
  ),
  
  tar_target(
    global_xgb_predictions,
    predict_xgb(model_xgb_intensity, cleaned_global_fires)
  ),
  
  tar_target(
    global_rf_confusion,
    caret::confusionMatrix(global_rf_predictions, cleaned_global_fires$intensity_class)
  ),
  
  tar_target(
    global_xgb_confusion,
    caret::confusionMatrix(global_xgb_predictions, cleaned_global_fires$intensity_class)
  ),
  
  # 8. DIAGNOSTICS — misclassification analysis on Australia test split
  tar_target(
    misclass_diagnostics,
    get_misclass_data(split_datasets$test, model_rf_intensity)
  ),
  
  tar_target(
    plot_misclass_density,
    plot_misclass_distributions(misclass_diagnostics)
  ),
  
  # 9. DIAGNOSTICS — misclassification analysis on Global data test split
  tar_target(
    global_misclass_diagnostics,
    get_misclass_data(cleaned_global_fires, model_rf_intensity)
  ),
  
  tar_target(
    plot_global_misclass_density,
    plot_misclass_distributions(global_misclass_diagnostics)
  )
)