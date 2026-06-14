#' Generate error tags for model predictions
#' @param test_data Dataframe containing true features and label 'intensity_class'
#' @param rf_model Trained random forest model object
get_misclass_data <- function(test_data, rf_model) {
  test_data %>%
    mutate(
      Predicted = predict(rf_model, newdata = test_data),
      Is_Correct = (intensity_class == Predicted),
      Error_Type = case_when(
        Is_Correct ~ "Correct",
        TRUE ~ paste0("Actual: ", intensity_class, " | Pred: ", Predicted)
      )
    )
}

#' Generate density plots to find where misclassifications are standardized
#' @param diagnostic_data Dataframe generated from get_misclass_data()
create_diagnostic_plots <- function(diagnostic_data) {
  
  # Create a density plot tracking bright_t31 distribution for errors vs correct predictions
  p1 <- ggplot(diagnostic_data, aes(x = bright_t31, fill = Is_Correct)) +
    geom_density(alpha = 0.4) +
    scale_fill_manual(values = c("FALSE" = "#ef4444", "TRUE" = "#10b981")) +
    theme_minimal() +
    labs(
      title = "Error Concentration across Brightness Temperature (t31)",
      subtitle = "Are misclassifications localized to specific thermal thresholds?",
      x = "Thermal Channel Brightness (t31)",
      y = "Density",
      fill = "Prediction Correct"
    )
  
  return(p1)
}