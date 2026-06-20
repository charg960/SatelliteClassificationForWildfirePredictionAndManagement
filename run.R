# runs pipeline and reviews results
source("R/functions.R") 
library(targets)
library(randomForest)
library(xgboost)
library(usethis)

tar_make()

test_data <- tar_read(split_datasets)$test
predictions <- predict(tar_read(model_rf_intensity), test_data)
caret::confusionMatrix(predictions, test_data$intensity_class)

#performance plots
tar_read(plot_ml_boundaries)
tar_read(plot_ml_predictions)

#heatmap
grid::grid.newpage()
grid::grid.draw(tar_read(plot_ml_heatmap)$gtable)

gert::git_status()
gert::git_init("/Users/litchar/Desktop/Satellite_Wildfire_Pipeline")


# 3. EXTRACT MODELS FOR DETAILED INSPECTION ----
rf_intensity <- tar_read(model_rf_intensity)

# Print out-of-bag error rates
print(rf_intensity)

# Plot feature importance rankings
randomForest::varImpPlot(rf_intensity, main = "What Signals Matter Most?")


# 4. PIPELINE HEALTH CHECK: View the visual graph network of your pipeline
tar_visnetwork()

# XGBoost evaluation
xgb_model <- tar_read(model_xgb_intensity)
xgb_preds <- predict_xgb(xgb_model, test_data)
caret::confusionMatrix(xgb_preds, test_data$intensity_class)

# 5. GLOBAL GENERALIZATION TEST
# These models were trained ONLY on Australian fire data. Here we test them
# on a completely different dataset: 2024 MODIS fire detections from 207
# countries worldwide. The Low/Medium/High thresholds were calculated once,
# from Australia's data only, and reused as-is for this global data (instead
# of being recalculated from scratch on each new dataset). This lets us see
# how the Australia-trained model and Australia-based thresholds hold up
# against real fire behavior elsewhere in the world.
tar_read(global_rf_confusion)
tar_read(global_xgb_confusion)

