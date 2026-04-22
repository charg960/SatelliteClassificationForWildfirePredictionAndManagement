library(tidyverse)
library(lubridate)
library(readxl)
library(caret)
library(randomForest)
library(dbscan)
library(factoextra)
library(cluster)
library(pheatmap)

fires <- read_excel(
  "/Users/litchar/Library/Mobile Documents/com~apple~Numbers/Documents/Fires_From_Space_Australia_Dataset.xlsx",
  skip = 1
)


fires <- fires %>%
  mutate(
    acq_time = sprintf("%04d", as.integer(acq_time)),
    time_formatted = paste0(substr(acq_time, 1, 2), ":", substr(acq_time, 3, 4), ":00"),
    timestamp = ymd_hms(paste(acq_date, time_formatted)),
    
    missing_radiance = is.na(frp) | is.na(bright_t31)
  ) %>%
  filter(frp >= 0, bright_t31 > 200, bright_t31 < 400) %>%
  arrange(latitude, longitude, timestamp) %>%
  group_by(round(latitude, 2), round(longitude, 2)) %>%
  arrange(timestamp, .by_group = TRUE) %>%
  mutate(
    local_time = timestamp + dhours(longitude / 15),
    hour_of_day = hour(local_time),
    diurnal_cycle = sin(2 * pi * hour_of_day / 24),
    time_since_last = as.numeric(difftime(timestamp, lag(timestamp), units = "hours"))
  ) %>%
  ungroup() %>%
  mutate(
    day_night = ifelse(hour_of_day >= 6 & hour_of_day <= 18, "Day", "Night"),
    
    intensity_class = case_when(
      frp < 30 ~ "Low",
      frp < 100 ~ "Medium",
      TRUE ~ "High"
    ),
     log_frp = log1p(frp)
  )



glimpse(fires)

fires %>%
  summarise(
    frp_missing = mean(is.na(frp)),
    brightness_missing = mean(is.na(bright_t31)),
    radiance_flag_rate = mean(missing_radiance)
  )



# CLASSIFICATION (MAIN SECTION)

fires_class <- fires %>%
  select(frp, bright_t31, hour_of_day, diurnal_cycle, time_since_last, day_night, intensity_class) %>%
  drop_na()

# ---- convert targets
fires_class$day_night <- as.factor(fires_class$day_night)
fires_class$intensity_class <- as.factor(fires_class$intensity_class)

set.seed(123)

train_index <- createDataPartition(fires_class$day_night, p = 0.7, list = FALSE)

train <- fires_class[train_index, ]
test <- fires_class[-train_index, ]



# MODEL 1: DAY vs NIGHT
day_model <- randomForest(day_night ~ frp + bright_t31 + hour_of_day + diurnal_cycle + time_since_last,
                          data = train)

day_pred <- predict(day_model, test)

confusionMatrix(day_pred, test$day_night)



# MODEL 2: FIRE INTENSITY
#intensity_model <- randomForest(intensity_class ~ frp + bright_t31 + hour_of_day + diurnal_cycle + time_since_last,
                           #     data = train)
intensity_model <- randomForest(
  intensity_class ~ bright_t31 + hour_of_day + diurnal_cycle + time_since_last,
  data = train
)
intensity_pred <- predict(intensity_model, test)

confusionMatrix(intensity_pred, test$intensity_class)
fires %>%
  count(intensity_class)



#Keep!!
# Day vs Night intensity differences: “Can satellites distinguish day vs night behavior?”
# FINDINGS: Fire Radiative Power differs between day and night detections, suggesting satellites capture meaningful diurnal variation in fire activity.
ggplot(fires, aes(day_night, frp)) + 
  geom_boxplot()

# Thermal signal separation: “Is there structure the model can learn?”
# FINDINGS: Relationship between time of day and fire intensity, separated by day/night classification.
ggplot(fires, aes(hour_of_day, frp, color = day_night)) +
  geom_point(alpha = 0.4)


#KEEP!!
# “What signals from space matter most?”
# FINDINGS: Thermal brightness and temporal features are the strongest predictors, indicating that satellites rely primarily on heat signatures rather than time patterns to classify fire intensity.
varImpPlot(intensity_model)


cm <- confusionMatrix(intensity_pred, test$intensity_class)$table

# KEEP!!!
#FINDINGS: “How many times did the model predict X when the true class was Y?”
#OR.....how often your model predicted each class correctly vs incorrectly
my_colors <- colorRampPalette(c(
  "#D7E4F4",  # light (low error)
  "#699F80",  # greenish
  "#E0CC67",  # yellow
  "#B24646"   # intense red (high error)
))(200)
cm_cat <- cut(
  cm,
  breaks = c(-Inf, 200, 600, Inf),
  labels = c("Low", "Medium", "High")
)
cm_num <- matrix(
  as.numeric(cm_cat),
  nrow = nrow(cm),
  ncol = ncol(cm),
  dimnames = dimnames(cm)
)
row_order <- order(rowMeans(cm_num))
col_order <- order(colMeans(cm_num))

cm_sorted <- cm[row_order, col_order]
pheatmap(
  cm_sorted,
  display_numbers = TRUE,
  cluster_rows = FALSE,
  cluster_cols = FALSE,
  color = my_colors,
  breaks = seq(min(cm), max(cm), length.out = 201)
)



#KEEP!!
#FINDINGS: This graph compares predicted fire intensity classes to actual classes, showing how often the model correctly or incorrectly classifies Low, Medium, and High intensity fires.
#This indicates LIMITATIONS in detecting extreme fire behavior from satellite-derived features alone.
ggplot(data.frame(pred = intensity_pred, actual = test$intensity_class),
       aes(actual, fill = pred)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c(
    "High" = "#B24646",
    "Medium" = "#E0CC67",
    "Low" = "#699F80"
  )) +
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "white"),
    plot.background = element_rect(fill = "white")
  )




library(rpart)
library(rpart.plot)
 
tree_model_real <- rpart(
  intensity_class ~ bright_t31 + hour_of_day + diurnal_cycle + time_since_last + day_night,
  data = train,
  method = "class",
  control = rpart.control(cp = 0.01, minbucket = 50)
)

rpart.plot(
  tree_model_real,
  type = 3,
  extra = 104,
  fallen.leaves = TRUE,
  box.palette = list("#D7E4F4", "#699F80", "#E0CC67")
)


ggplot(fires, aes(day_night, frp, fill = day_night)) +
  geom_violin(trim = FALSE, alpha = 0.7) +
  geom_jitter(width = 0.1, alpha = 0.3, color = "gray30") +
  scale_fill_manual(values = c("D" = "#E0CC67", "N" = "#699F80")) +
  labs(title = "Distribution of FRP by Day vs Night") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none")

loadings <- pca$rotation[, 1:2]
loadings

# PCA on predictors
pca <- prcomp(train[, c("bright_t31", "hour_of_day", "diurnal_cycle", "time_since_last")],
              scale. = TRUE)
train_pca <- as.data.frame(pca$x[, 1:2])

train_pca$intensity_class <- train$intensity_class
rf_pca <- randomForest(
  intensity_class ~ PC1 + PC2,
  data = train_pca,
  ntree = 300
)

tree_model_real <- rpart(
  intensity_class ~ bright_t31 + hour_of_day + diurnal_cycle + time_since_last + day_night,
  data = train_tree,
  method = "class",
  control = rpart.control(
    cp = 0.0005,       # allow more splits
    minsplit = 20,     
    minbucket = 7,    
    maxdepth = 3      
  )
)

rpart.plot(
  tree_model_real,
  type = 3,
  extra = 104,
  fallen.leaves = TRUE,
  box.palette = list("#D7E4F4", "#699F80", "#E0CC67")
)


ggplot() +
  geom_tile(data = grid, aes(PC1, PC2, fill = pred), alpha = 0.55) +
  geom_point(data = train_pca, aes(PC1, PC2, color = intensity_class), size = 1.8) +
  scale_fill_manual(values = c(
    "Low" = "#80CDC1",
    "Medium" = "#F1A340",
    "High" = "#B2182B"
  )) +
  scale_color_manual(values = c(
    "Low" = "#01665E",
    "Medium" = "#E66101",
    "High" = "#B2182B"
  )) +
  labs(
    title = "Intensity Classification Boundaries",
    subtitle = "Random Forest on PCA‑Reduced Features",
    x = "Thermal Intensity",   
    y = "Diurnal Behavior",  
    fill = "Predicted Class",
    color = "Actual Class"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold"),
    legend.position = "right"
  )

 

#Binary logistic regression!!: MODEL Day vs Night 
day_glm <- glm(
  day_night ~ frp + bright_t31 + time_since_last,
  data = train,
  family = binomial
)
day_prob <- predict(day_glm, test, type = "response")
day_pred_glm <- ifelse(day_prob > 0.5, "Day", "Night") %>% as.factor()
confusionMatrix(day_pred_glm, test$day_night)
#Multivariate logistic regression!!: MODEL Fire Intensity
library(nnet)
intensity_glm <- multinom(
  intensity_class ~ bright_t31 + hour_of_day + diurnal_cycle + time_since_last,
  data = train
)

intensity_pred_glm <- predict(intensity_glm, test)

confusionMatrix(intensity_pred_glm, test$intensity_class)
summary(intensity_glm)

# RANDOM FOREST: DAY vs NIGHT
set.seed(123)

rf_day <- randomForest(
  day_night ~ frp + bright_t31 + hour_of_day + diurnal_cycle + time_since_last,
  data = train,
  ntree = 500,
  mtry = 3,
  importance = TRUE
)

rf_day_pred <- predict(rf_day, test)

confusionMatrix(rf_day_pred, test$day_night)

# Variable importance
varImpPlot(rf_day, main = "Variable Importance: Day vs Night")


# RANDOM FOREST: FIRE INTENSITY
set.seed(123)

rf_intensity <- randomForest(
  intensity_class ~ bright_t31 + hour_of_day + diurnal_cycle + time_since_last,
  data = train,
  ntree = 500,
  mtry = 2,
  importance = TRUE
)

rf_intensity_pred <- predict(rf_intensity, test)

confusionMatrix(rf_intensity_pred, test$intensity_class)

# Variable importance
varImpPlot(rf_intensity, main = "Variable Importance: Fire Intensity")
