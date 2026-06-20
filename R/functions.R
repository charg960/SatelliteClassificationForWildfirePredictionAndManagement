# ---- DATA PROCESSING ----
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

plot_misclass_distributions <- function(diagnostic_data) {
  ggplot(diagnostic_data, aes(x = bright_t31, fill = Is_Correct)) +
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
}

load_and_clean_fires <- function(filepath, frp_low_cutoff = NULL, frp_high_cutoff = NULL) {
  if (grepl("\\.xlsx$", filepath)) {
    fires <- readxl::read_excel(filepath, skip=1)
  } else if (grepl("\\.csv$", filepath)) {
    fires <- readr::read_csv(filepath)
  } else {
    stop("Unsupported file type: ", filepath)
  }

  if (is.null(frp_low_cutoff)) frp_low_cutoff <- quantile(fires$frp, 0.33, na.rm=TRUE)
  if (is.null(frp_high_cutoff)) frp_high_cutoff <- quantile(fires$frp, 0.67, na.rm=TRUE)
  
  fires %>%
    mutate(
      acq_time = sprintf("%04d", as.integer(acq_time)),
      time_formatted = paste0(substr(acq_time, 1, 2), ":", substr(acq_time, 3, 4), ":00"),
      timestamp = ymd_hms(paste(acq_date, time_formatted)),
      missing_radiance = is.na(frp) | is.na(bright_t31),
      pixel_area = scan * track,          
      frp_density = frp / (scan * track),
      confidence_num = as.numeric(confidence)
    ) %>%
    filter(frp >= 0, bright_t31 > 200, bright_t31 < 400) %>%
    
    # ===== TIMEZONE / DIURNAL CALCULATION — done ONCE, ungrouped, on the whole dataset =====
  mutate(
    tz_name = lutz::tz_lookup_coords(lat = latitude, lon = longitude, method = "fast")
  ) %>%
    group_by(tz_name) %>%
    mutate(
      local_time = lubridate::with_tz(timestamp, tzone = tz_name[1])
    ) %>%
    ungroup() %>%
    mutate(
      hour_of_day = hour(local_time),
      diurnal_cycle = sin(2 * pi * hour_of_day / 24)
    ) %>%
    
    # ===== SPATIAL GROUPING #1: ~1.1km precision =====
  # Used ONLY to compare consecutive detections at essentially the same location,
  # so time_since_last reflects the gap since the last detection at that SAME spot.
  # Resolution: round(lat/lon, 2) ≈ 0.01° ≈ 1.1km grid cells.
  # NOTE: originally implemented via group_by(), which was prohibitively slow
  # at global scale (~3.5M distinct groups out of 4.85M rows — nearly 1:1,
  # so grouping overhead dominated). Replaced with: sort once by location+time,
  # then compare each row to the row immediately before it — mathematically
  # equivalent, but avoids group_by() entirely.
  mutate(
    lat_round = round(latitude, 2),
    lon_round = round(longitude, 2)
  ) %>%
    arrange(lat_round, lon_round, timestamp) %>%
    mutate(
      prev_lat_round = lag(lat_round),
      prev_lon_round = lag(lon_round),
      prev_timestamp = lag(timestamp),
      same_location = (lat_round == prev_lat_round) & (lon_round == prev_lon_round),
      time_since_last = if_else(
        same_location,
        as.numeric(difftime(timestamp, prev_timestamp, units = "hours")),
        NA_real_
      )
      # ^ time_since_last feeds: train_rf_day(), train_rf_intensity(),
      #   train_decision_tree(), train_xgb_intensity(), compute_pca()
    ) %>%
    select(-lat_round, -lon_round, -prev_lat_round, -prev_lon_round, -prev_timestamp, -same_location) %>%
    
    # ===== SPATIAL GROUPING #2: coarse, ~11km precision, 2-hour time bins =====
  # Used to find FIRE NEIGHBORS — other detections nearby in space AND time,
  # to estimate local fire spread/clustering (not just one pixel's own reading).
  # Resolution: round(lat/lon, 1) ≈ 0.1° ≈ 11km grid cells, 7200-sec (2hr) time bins.
  # NOTE: this is a genuine group-wise aggregation (mean + count per group), so
  # unlike Spatial Grouping #1, we can't avoid grouping entirely. dplyr::group_by()
  # was prohibitively slow at global scale; data.table's grouped aggregation
  # handles the same high-group-cardinality case in seconds instead of minutes.
  {
    dt <- data.table::as.data.table(.)
    dt[, lat_bin := round(latitude, 1)]
    dt[, lon_bin := round(longitude / cos(latitude * pi / 180), 1)]
    dt[, time_bin := floor(as.numeric(timestamp) / 7200)]
    dt[, neighbor_mean_frp := mean(frp, na.rm = TRUE), by = .(lat_bin, lon_bin, time_bin)]
    # ^ neighbor_mean_frp feeds: train_decision_tree() ONLY
    #   (NOT used by train_rf_intensity() or train_xgb_intensity() currently)
    dt[, neighbor_count := .N, by = .(lat_bin, lon_bin, time_bin)]
    # ^ neighbor_count feeds: train_decision_tree() ONLY
    #   (NOT used by train_rf_intensity() or train_xgb_intensity() currently)
    dt[, c("lat_bin", "lon_bin", "time_bin") := NULL]
    as.data.frame(dt)
  } %>%
    
    mutate(
      day_night = factor(ifelse(hour_of_day >= 6 & hour_of_day <= 18, "Day", "Night")),
      satellite = factor(satellite),
      type = factor(type),
      intensity_class = factor(case_when(
        frp < frp_low_cutoff  ~ "Low",
        frp < frp_high_cutoff ~ "Medium",
        TRUE ~ "High"
      ), levels = c("Low", "Medium", "High")),
      # ^ intensity_class is the TARGET LABEL for:
      #   train_rf_intensity(), train_xgb_intensity(), train_decision_tree(),
      #   train_rf_pca(), and every plot_* / diagnostic function downstream
      log_frp = log1p(frp)
    ) %>%
    drop_na(frp, bright_t31, hour_of_day, diurnal_cycle, time_since_last, 
            day_night, intensity_class, confidence_num, pixel_area, frp_density)
}

split_wildfire_data <- function(data, p = 0.7, seed = 123) {
  set.seed(seed)
  train_index <- caret::createDataPartition(data$day_night, p = p, list = FALSE)
  list(
    train = data[train_index, ],
    test  = data[-train_index, ]
  )
}

# ---- VISUAL EDAs ----
plot_diurnal_boxplots <- function(data) {
  ggplot(data, aes(day_night, frp, fill = day_night)) + 
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_jitter(width = 0.1, alpha = 0.3, color = "gray30") +
    scale_fill_manual(values = c("Day" = "#E0CC67", "Night" = "#699F80")) +
    labs(title = "Distribution of FRP by Day vs Night") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none")
}

plot_thermal_signals <- function(data) {
  ggplot(data, aes(hour_of_day, frp, color = day_night)) +
    geom_point(alpha = 0.4) +
    theme_minimal() +
    labs(title = "Thermal Signal Separation")
}



# ---- MACHINE LEARNING MODELS ----
train_rf_day <- function(train_data) {
  set.seed(123)
  randomForest(day_night ~ frp + bright_t31 + hour_of_day + diurnal_cycle + time_since_last,
               data = train_data, ntree = 500, mtry = 3, importance = TRUE)
}

train_decision_tree <- function(train_data) {
  rpart(
    formula = intensity_class ~ bright_t31 + hour_of_day + diurnal_cycle + 
      time_since_last + pixel_area + frp_density + 
      confidence_num + neighbor_mean_frp + neighbor_count,
    data = train_data, 
    method = "class",
    control = rpart.control(
      cp = 0.01,        # Standard starting place for pruning
      maxdepth = 6,     # Allows deeper interactions (such as bright_t31 nested inside time_of_day)
      minsplit = 20, 
      minbucket = 7
    )
  )
}

compute_pca <- function(train_data) {
  prcomp(train_data[, c("bright_t31", "hour_of_day", "diurnal_cycle", "time_since_last")], scale. = TRUE)
}

train_rf_intensity <- function(train_data) {
  class_counts <- table(train_data$intensity_class)
  # Weight by inverse frequency!!
  weights <- 1 / sqrt(class_counts[train_data$intensity_class])
  
  randomForest(
    formula = intensity_class ~ bright_t31 + hour_of_day + diurnal_cycle + 
      time_since_last + pixel_area + frp_density + confidence_num,
    data = train_data,
    ntree = 500,
    mtry = 4,
    importance = TRUE,
    classwt = 1 / sqrt(class_counts)  # softer than sampsize
  )
}

train_rf_pca <- function(pca_obj, train_data) {
  train_pca <- as.data.frame(pca_obj$x[, 1:2])
  train_pca$intensity_class <- train_data$intensity_class
  randomForest(intensity_class ~ PC1 + PC2, data = train_pca, ntree = 300)
}

# ---- PIPELINE VISUAL RESULTS ----
plot_intensity_predictions <- function(model, test_data) {
  preds <- predict(model, test_data)
  ggplot(data.frame(pred = preds, actual = test_data$intensity_class), aes(actual, fill = pred)) +
    geom_bar(position = "dodge") +
    scale_fill_manual(values = c("High" = "#B24646", "Medium" = "#E0CC67", "Low" = "#699F80")) +
    theme_minimal(base_size = 14)
}

plot_pca_boundaries <- function(pca_obj, rf_pca_model, train_data) {
  train_pca <- as.data.frame(pca_obj$x[, 1:2])
  train_pca$intensity_class <- train_data$intensity_class
  
  grid_range <- expand.grid(
    PC1 = seq(min(train_pca$PC1) - 1, max(train_pca$PC1) + 1, length.out = 100),
    PC2 = seq(min(train_pca$PC2) - 1, max(train_pca$PC2) + 1, length.out = 100)
  )
  grid_range$pred <- predict(rf_pca_model, grid_range)
  
  ggplot() +
    geom_tile(data = grid_range, aes(PC1, PC2, fill = pred), alpha = 0.55) +
    geom_point(data = train_pca, aes(PC1, PC2, color = intensity_class), size = 1.8) +
    scale_fill_manual(values = c("Low" = "#80CDC1", "Medium" = "#F1A340", "High" = "#B2182B")) +
    scale_color_manual(values = c("Low" = "#01665E", "Medium" = "#E66101", "High" = "#B2182B")) +
    theme_minimal(base_size = 14)
}

plot_pheatmap_cm <- function(model, test_data) {
  cm <- caret::confusionMatrix(predict(model, test_data), test_data$intensity_class)$table
  my_colors <- colorRampPalette(c("#D7E4F4", "#699F80", "#E0CC67", "#B24646"))(200)
  pheatmap(cm, display_numbers = TRUE, cluster_rows = FALSE, cluster_cols = FALSE,
           color = my_colors, breaks = seq(min(cm), max(cm), length.out = 201), silent = TRUE)
}

train_xgb_intensity <- function(train_data) {
  library(xgboost)
  
  label_map <- c("Low" = 0, "Medium" = 1, "High" = 2)
  labels <- label_map[as.character(train_data$intensity_class)]
  
  feature_cols <- c("bright_t31", "hour_of_day", "diurnal_cycle", 
                    "time_since_last", "pixel_area", "frp_density",
                    "confidence_num")
  mat <- as.matrix(train_data[, feature_cols])
  
  dtrain <- xgb.DMatrix(data = mat, label = labels)
  
  xgb.train(
    data = dtrain,
    nrounds = 400,
    params = list(
      objective = "multi:softmax",
      num_class = 3,
      max_depth = 6,
      learning_rate = 0.05,
      subsample = 0.8,
      colsample_bytree = 0.8
    ),
    verbose = 0
  )
}

predict_xgb <- function(xgb_model, new_data) {
  feature_cols <- c("bright_t31", "hour_of_day", "diurnal_cycle",
                    "time_since_last", "pixel_area", "frp_density",
                    "confidence_num")  # <-- ADD this, was missing
  mat <- as.matrix(new_data[, feature_cols])
  dtest <- xgb.DMatrix(data = mat)  # <-- also fix typo: was "dtест" (cyrillic т)
  preds <- predict(xgb_model, dtest)
  factor(c("Low", "Medium", "High")[preds + 1], levels = c("Low", "Medium", "High"))
}

# Re-draw the cached test set evaluation heatmap
grid::grid.newpage()
grid::grid.draw(targets::tar_read(plot_ml_heatmap)$gtable)