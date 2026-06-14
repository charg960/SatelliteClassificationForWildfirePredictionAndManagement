# runs pipeline and reviews results
source("R/functions.R") 
library(targets)
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
gert::git_progress()
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
