#!/usr/bin/Rscript 
library(keras, quietly = TRUE)
use_implementation("keras")
use_backend("plaidml")
library(testit)
assert("Using Plaidml", k_backend() == "plaidml")
library(kerasR)
library(caret)
library(randomForest)
library(dplyr) ##be careful with explain function
library(ggplot2)
library(ggforce)
library(tidyquant)
library(scales)
library(corrr)
library(vip)
library(lime)
library(DALEX)
library(yardstick)
library(filenamer)
library(reticulate)
library(ceterisParibus)
library(ggthemr)
ggthemr("light")
library(readr)
library(readxl, quietly = TRUE)
library(robustbase)
library(data.table)
library(missRanger)
library(plotly)
library(lazyeval)
library(likert)
library(rworldmap)
library(ggrepel)
library(stringi)
library(ggthemes)
library(extrafont)
library(ingredients)
library(flashlight)
library(janitor)
library(stringr)
library(forcats)
library(gridExtra)
library(grid)
options(digits = 16)
INT_NA <- -9


#change only these
imputed_unseen_filename <-  "/research/dataset/world_values/unethical/imputed_unseen_2019-11-25.rds"
load("/research/dataset/world_values/unethical/hyperparameter_environment_2020-03-17.RData")

#Prepare some image filenames
fivsr_image <- paste(basedatadir, "image_FIVSR_C.png", sep='/')
fivs_image <- paste(basedatadir, "image_FIV_S.png", sep='/')
fih_image <- paste(basedatadir, "image-FIH.png", sep='/')
fi_image <- paste(basedatadir, "image_FI.png", sep='/')
mck_image <- paste(basedatadir, "image_MCK.png", sep='/')
mcl_image <- paste(basedatadir, "image_MCL.png", sep='/')
vic_image <- paste(basedatadir, "image_VIC.png", sep='/')
cp_image <- paste(basedatadir, "image_CP.png", sep='/')
pdp_v1_image <- paste(basedatadir, "image_pdp_top_v1.png", sep='/')
pdp_v2_image  <- paste(basedatadir, "image_pdp_top_v2.png", sep='/')
pdp_v3_image  <- paste(basedatadir, "image_pdp_top_v3.png", sep='/')
pdp_v4_image  <- paste(basedatadir, "image_pdp_top_v4.png", sep='/')
spc_image  <- paste(basedatadir, "image_SPC.png", sep='/')
spsr_image  <- paste(basedatadir, "image_SPSR.png", sep='/')
country_file <- paste(basedatadir, "country_predictors.csv", sep='/')
wave_file <- paste(basedatadir, "wave_predictors.csv", sep='/')
subregion_file <- paste(basedatadir, "subregion_predictors.csv", sep='/')
confusion_image <- paste(basedatadir, "image_confusion.png", sep='/')
world_file <- paste(basedatadir, "world_predictors.csv", sep='/')
wmc_image <- paste(basedatadir, "WMC.png", sep='/')
FTA_image <- paste(basedatadir, "FTA.png", sep='/')
SR_predictability <- paste(basedatadir, "subregion_predictability.csv", sep='/')
CY_predictability <- paste(basedatadir, "country_predictability.csv", sep='/')
absolute_interaction <- paste(basedatadir, "absolute_interaction.png", sep='/')
pairwise_interaction <- paste(basedatadir, "pairwise_interaction.png", sep='/')

Sys.setenv(PLAIDML_DEVICE_IDS="opencl_nvidia_geforce_gtx_1070.0")

model_keras <- load_model_hdf5(model_file, compile=FALSE)
#load_model_weights_hdf5(model_keras, modelfile)


#get the unseen data
df7.test <- readRDS(imputed_unseen_filename) %>%
  create_label() %>%
  select(c(final_keep_vars, regress_var)) %>%
  my_shuffle()

test_input_matrix <- predict(preProcessModel, dplyr::select(df7.test, -c(regress_var))) %>%
  as.matrix()

test_output_matrix <- as.matrix(dplyr::select(df7.test, c(regress_var)))

#get confusion matrix
Y_test_hat <- keras_predict(model_keras, test_input_matrix, verbose=2)
table(test_output_matrix, round(Y_test_hat))
print(paste("Keras Accuracy:", mean(test_output_matrix == as.numeric(round(Y_test_hat)))))


#
# Plot confusion matrix for report
#
options(digits = 4)
cfm <- confusionMatrix(as.factor(round(as.vector(Y_test_hat))), as.factor(as.vector(test_output_matrix)), positive = "1")
cfm
ggplotConfusionMatrix <- function(m){
  mytitle <- paste("Accuracy", percent_format(accuracy =.01)(m$overall[1]),
                   "Kappa", percent_format(accuracy = .01)(m$overall[2]))
  p <-
    ggplot(data = as.data.frame(m$table) ,
           aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    geom_text(aes(x = Reference, y = Prediction, label = Freq)) +
    theme(legend.position = "none", axis.title = element_text(size = 10), plot.title = element_text(size = 10)) +
    ggtitle(mytitle) +  xlab("Actual") + ylab("Predicted")
  return(p)
}
ggplotConfusionMatrix(cfm)
ggsave(confusion_image, width = 5, height = 2, units =  "in", dpi=300)

#
# Check performance
#
# Predicted Class
yhat_keras_class_vec <- predict_classes(object = model_keras, x = as.matrix(test_input_matrix)) %>%
  as.vector()

# Predicted Class Probability
yhat_keras_prob_vec  <- predict_proba(object = model_keras, x = as.matrix(test_input_matrix)) %>%
  as.vector()

# Format test data and predictions for yardstick metrics

estimates_keras_tbl <- tibble(
  keras_sample_id  = 1:nrow(test_input_matrix),
  keras_truth      = as.factor(test_output_matrix) %>% fct_recode(yes = "1", no = "0"),
  keras_estimate   = as.factor(yhat_keras_class_vec) %>% fct_recode(yes = "1", no = "0"),
  keras_class_prob = yhat_keras_prob_vec
)

estimates_keras_tbl$correct <- ifelse(estimates_keras_tbl$keras_truth == estimates_keras_tbl$keras_estimate, "correct", "wrong")
estimates_keras_tbl$Caught <- ifelse((estimates_keras_tbl$keras_truth) == "yes" & (estimates_keras_tbl$keras_estimate == "yes"), "Yes", "No")
estimates_keras_tbl$Situation  <-
  case_when(
      (estimates_keras_tbl$keras_truth == "yes") & (estimates_keras_tbl$keras_estimate == "yes") ~ "Incarcerated",
      (estimates_keras_tbl$keras_truth == "yes") & (estimates_keras_tbl$keras_estimate == "no") ~ "Escaped",
      (estimates_keras_tbl$keras_truth == "no") & (estimates_keras_tbl$keras_estimate == "yes") ~ "Victimized",
      (estimates_keras_tbl$keras_truth == "no") & (estimates_keras_tbl$keras_estimate == "no") ~ "Ideal_Citizen"
  )


# tabulate values againt countries
#
country_ISO_values <- read_csv("~/Work/Experiments/dataset/world_values/ISO_country_region_AS.csv") %>%
  rename(country = name, ISO3 = `alpha-3`,  subregion = `sub-region`) %>%
  mutate(countryID = as.integer(`country-code`)) %>%
  mutate(subregionID = as.integer(`sub-region-code`)) %>%
  dplyr::select(c("country", "region", "subregion", "countryID", "subregionID")) %>%
  mutate(country = stri_replace_all_charclass(paste("CY_",country,sep=""), "\\p{WHITE_SPACE}", ""))
#
# Reencode country
#
temp <- df7.test %>%
  dplyr::select(matches("^s003_"))
estimates_keras_tbl$country <- names(temp)[max.col(temp)]

#
# Reencode subregion
#
temp <- df7.test %>%
  dplyr::select(matches("^sub_region_code_"))
estimates_keras_tbl$subregion <- names(temp)[max.col(temp)]

#
# Reencode wave
#
temp <- df7.test %>%
  dplyr::select(matches("^s002_"))
estimates_keras_tbl$wave <- names(temp)[max.col(temp)]

#
# Begin code for plots
country_summary <- estimates_keras_tbl %>%
  group_by(country)  %>%
  summarize(n_escaped = sum(Situation == "Escaped"),
            n_incarcerated = sum(Situation == "Incarcerated"),
            n_victimized = sum(Situation == "Victimized"),
            n_ideal_citizen = sum(Situation == "Ideal_Citizen")) %>%
  mutate(n_total = n_escaped + n_incarcerated + n_victimized + n_ideal_citizen) %>%
  mutate(p_escaped = n_escaped/n_total) %>%
  mutate(p_incarcerated = n_incarcerated/n_total) %>%
  mutate(p_victimized = n_victimized/n_total) %>%
  mutate(p_ideal_citizen = n_ideal_citizen/n_total) %>%
  arrange(p_escaped)

temp_index <- match(as.integer(substring(country_summary$country, 6)), country_ISO_values$countryID)
Country <- country_ISO_values[temp_index, 'country']$country %>%
  substring(4)
p_victimized <- country_summary$p_victimized
p_escaped <- country_summary$p_escaped
p_ideal_citizen <- country_summary$p_ideal_citizen
p_incarcerated <- country_summary$p_incarcerated
write.csv(country_summary, CY_predictability)

plot_ly(country_summary, 
        x = ~p_ideal_citizen, y = ~Country, type = 'bar', orientation = 'h', name = 'p_ideal_citizen',
        marker = list(color = 'rgba(243, 208, 62, 1)', line = list(color = 'rgba(58, 71, 80, 0.5)', width = 0.4)) ) %>%
  add_trace(x = ~p_victimized, marker = list(color = 'rgba(119, 197, 213, 1)'), name = 'p_victimized') %>%
  add_trace(x = ~p_incarcerated, marker = list(color = 'rgba(237, 139, 0, 1)'), name = 'p_incarcerated') %>% 
  add_trace(x = ~p_escaped, marker = list(color = 'rgba(209, 65, 36, 1)') , name = 'p_escaped') %>%
  
  layout(xaxis = list(title = "",
                      showgrid = FALSE,
                      showline = FALSE,
                      showticklabels = FALSE,
                      zeroline = FALSE,            
                      domain = c(0.15, 1) ),
         
         yaxis = list(title = "",
                      showgrid = FALSE,
                      showline = FALSE,
                      showticklabels = FALSE,
                      zeroline = FALSE,categoryorder = 'trace'),
         barmode = 'stack',
         paper_bgcolor = 'rgb(248, 248, 255)', plot_bgcolor = 'rgb(248, 248, 255)',
         margin = list(l = 10, r = 10, t = 10, b = 10),
         showlegend = TRUE ,legend = list(traceorder = "normal")) %>%
  # labeling the y-axis
  add_annotations(xref = 'paper', yref = 'y', x = 0.14, y = Country,
                  xanchor = 'right',
                  text = Country,
                  font = list(family = 'Arial', size = 8,
                              color = 'rgb(0, 0, 0)'),
                  showarrow = FALSE, align = 'right')
  # labeling the first Likert scale (on the top)
  # add_annotations(xref = 'x', yref = 'paper',
  #                 x = c(5 / 2, 5 + 44 / 2, 5 + 44 + 37 / 2, 5 + 44 + 37 + 14 / 2),
  #                 y = 1.15,
  #                 text = c('p_victimized','p_ideal_citizen','p_incarcerated','p_escaped'),
  #                 font = list(family = 'Arial', size = 12,
  #                             color = 'rgb(67, 67, 67)'),
  #                 showarrow = FALSE)

#
#  Save this image manually using  screenshot
#

# Plot a graph of subregion predictions
#
subregion_summary <- estimates_keras_tbl %>%
  group_by(subregion)  %>%
  summarize(n_escaped = sum(Situation == "Escaped"),
            n_incarcerated = sum(Situation == "Incarcerated"),
            n_victimized = sum(Situation == "Victimized"),
            n_ideal_citizen = sum(Situation == "Ideal_Citizen")) %>%
  mutate(n_total = n_escaped + n_incarcerated + n_victimized + n_ideal_citizen) %>%
  mutate(p_escaped = n_escaped/n_total) %>%
  mutate(p_incarcerated = n_incarcerated/n_total) %>%
  mutate(p_victimized = n_victimized/n_total) %>%
  mutate(p_ideal_citizen = n_ideal_citizen/n_total) %>%
  arrange(p_escaped)

temp_index <- match(as.integer(substring(subregion_summary$subregion, 17)), country_ISO_values$subregionID)
SubRegion <- country_ISO_values[temp_index, 'subregion']$subregion
p_victimized <- subregion_summary$p_victimized
p_escaped <- subregion_summary$p_escaped
p_ideal_citizen <- subregion_summary$p_ideal_citizen
p_incarcerated <- subregion_summary$p_incarcerated
write.csv(subregion_summary, SR_predictability)

plot_ly(subregion_summary, x = ~p_ideal_citizen, y = ~SubRegion, type = 'bar', width = 1, orientation = 'h', name = 'p_ideal_citizen',
             marker = list(color = 'rgba(243, 208, 62, 1)',
                           line = list(color = 'rgba(58, 71, 80, 0.5)', width = 0.4)) ) %>%
  add_trace(x = ~p_victimized, marker = list(color = 'rgba(119, 197, 213, 1)'), name = 'p_victimized') %>%
  add_trace(x = ~p_incarcerated, marker = list(color = 'rgba(237, 139, 0, 1)'), name = 'p_incarcerated') %>% 
  add_trace(x = ~p_escaped, marker = list(color = 'rgba(209, 65, 36, 1)') , name = 'p_escaped') %>%
  layout(autosize = F, width = 1370, height = 400,
        xaxis = list(title = "",
                      showgrid = FALSE,
                      showline = FALSE,
                      showticklabels = FALSE,
                      zeroline = FALSE,            
                      domain = c(0.15, 1) ),
         yaxis = list(title = "",
                      showgrid = FALSE,
                      showline = FALSE,
                      showticklabels = FALSE,
                      zeroline = FALSE,categoryorder = 'trace'),
         barmode = 'stack',
         paper_bgcolor = 'rgb(248, 248, 255)', plot_bgcolor = 'rgb(248, 248, 255)',
         margin = list(l = 60, r = 600, t = 10, b = 10),
         showlegend = TRUE ,legend = list(traceorder = "normal")) %>%
  # labeling the y-axis
  add_annotations(xref = 'paper', yref = 'y', x = 0.14, y = SubRegion,
                  xanchor = 'right',
                  text = SubRegion,
                  font = list(family = 'PTSans-Narrow', size = 14,
                              color = 'rgb(0, 0, 0)'),
                  showarrow = FALSE, align = 'right')

#
# Plot a graph of wave predictions
#
wave_summary <- estimates_keras_tbl %>%
  group_by(wave)  %>%
  summarize(n_escaped = sum(Situation == "Escaped"),
            n_incarcerated = sum(Situation == "Incarcerated"),
            n_victimized = sum(Situation == "Victimized"),
            n_ideal_citizen = sum(Situation == "Ideal_Citizen")) %>%
  mutate(n_total = n_escaped + n_incarcerated + n_victimized + n_ideal_citizen) %>%
  mutate(p_escaped = n_escaped/n_total) %>%
  mutate(p_incarcerated = n_incarcerated/n_total) %>%
  mutate(p_victimized = n_victimized/n_total) %>%
  mutate(p_ideal_citizen = n_ideal_citizen/n_total) %>%
  arrange(desc(wave))

Wave <- paste("Wave", substring(wave_summary$wave, 6), " ")
p_victimized <- wave_summary$p_victimized
p_escaped <- wave_summary$p_escaped
p_ideal_citizen <- wave_summary$p_ideal_citizen
p_incarcerated <- wave_summary$p_incarcerated

plot_ly(wave_summary, x = ~p_ideal_citizen, y = ~Wave, type = 'bar', width = 1, orientation = 'h', name = 'p_ideal_citizen',
        marker = list(color = 'rgba(243, 208, 62, 1)',
                      line = list(color = 'rgba(58, 71, 80, 0.5)', width = 0.4)) ) %>%
  add_trace(x = ~p_victimized, marker = list(color = 'rgba(119, 197, 213, 1)'), name = 'p_victimized') %>%
  add_trace(x = ~p_incarcerated, marker = list(color = 'rgba(237, 139, 0, 1)'), name = 'p_incarcerated') %>% 
  add_trace(x = ~p_escaped, marker = list(color = 'rgba(209, 65, 36, 1)') , name = 'p_escaped') %>%
  layout(autosize = F, width = 430, height = 150, 
        xaxis = list(title = "",
                      showgrid = FALSE,
                      showline = FALSE,
                      showticklabels = FALSE,
                      zeroline = FALSE,            
                      domain = c(0.15, 1) ),
         yaxis = list(title = "",
                      showgrid = FALSE,
                      showline = FALSE,
                      showticklabels = FALSE,
                      zeroline = FALSE,categoryorder = 'trace'),
         barmode = 'stack',
         paper_bgcolor = 'rgb(248, 248, 255)', plot_bgcolor = 'rgb(248, 248, 255)',
         margin = list(l = 60, r = 10, t = 10, b = 10),
         showlegend = TRUE ,legend = list(traceorder = "normal")) %>%
  # labeling the y-axis
  add_annotations(xref = 'paper', yref = 'y', x = 0.14, y = Wave,
                  xanchor = 'right',
                  text = Wave,
                  font = list(family = 'Arial', size = 12,
                              color = 'rgb(0, 0, 0)'),
                  showarrow = FALSE, align = 'right')
#
#

# generate predictability across waves
wave_columns <- dplyr::select(data.frame(test_input_matrix), matches("s002_"))
wave_names <- names(wave_columns)
wave_data <- cbind(as_tibble(test_input_matrix), as_tibble(test_output_matrix))
for (wave in wave_names) {
  wave_frame <- wave_data %>%
    filter_(interp(~ var == 1, var = as.name(wave)))
  print(paste("working on", wave, "Got Observations", nrow(wave_frame)))
  wave_input_matrix <- select(wave_frame, -c(regress_var)) %>% as.matrix()
  wave_output_matrix <- select(wave_frame, c(regress_var)) %>% as.matrix()
  #get confusion matrix
  Y_test_hat <- keras_predict(model_keras, wave_input_matrix, verbose=2)
  table(wave_output_matrix, round(Y_test_hat))
  print(paste("Keras Accuracy:", mean(wave_output_matrix == as.numeric(round(Y_test_hat)))))
}

# generate predictability across subregions
subregion_columns <- dplyr::select(data.frame(test_input_matrix), matches("sub_region_code_"))
subregion_names <- names(subregion_columns)
subregion_data <- cbind(as_tibble(test_input_matrix), as_tibble(test_output_matrix))
for (subregion in subregion_names) {
  subregion_frame <- subregion_data %>%
    filter_(interp(~ var == 1, var = as.name(subregion)))
  print(paste("working on", subregion, "Got Observations", nrow(subregion_frame)))
  subregion_input_matrix <- select(subregion_frame, -c(regress_var)) %>% as.matrix()
  subregion_output_matrix <- select(subregion_frame, c(regress_var)) %>% as.matrix()
  #get confusion matrix
  Y_test_hat <- keras_predict(model_keras, subregion_input_matrix, verbose=2)
  table(subregion_output_matrix, round(Y_test_hat))
  print(paste("Keras Accuracy:", mean(subregion_output_matrix == as.numeric(round(Y_test_hat)))))
}

# generate predictability across countries
country_columns <- dplyr::select(data.frame(test_input_matrix), matches("s003_"))
country_names <- names(country_columns)
country_data <- cbind(as_tibble(test_input_matrix), as_tibble(test_output_matrix))
for (country in country_names) {
  country_frame <- country_data %>%
    filter_(interp(~ var == 1, var = as.name(country)))
  print(paste("working on", country, "Got Observations", nrow(country_frame)))
  country_input_matrix <- select(country_frame, -c(regress_var)) %>% as.matrix()
  country_output_matrix <- select(country_frame, c(regress_var)) %>% as.matrix()
  #get confusion matrix
  Y_test_hat <- keras_predict(model_keras, country_input_matrix, verbose=2)
  table(country_output_matrix, round(Y_test_hat))
  print(paste("Keras Accuracy:", mean(country_output_matrix == as.numeric(round(Y_test_hat)))))
}

#heatmap
#
#

#
#load country table
#
iso_country_codes <- read_csv("~/Work/Experiments/dataset/world_values/ISO_country_region_AS.csv")
#
# tabulate values againt countries
#

country_values <- iso_country_codes %>%
  rename(country = name, ISO3 = `alpha-3`, countryID = `country-code`, subregion = `sub-region`) %>%
  dplyr::select(c("ISO3","country", "region", "subregion", "countryID"))  %>%
  mutate(country = stri_replace_all_charclass(paste("CY_",country,sep=""), "\\p{WHITE_SPACE}", "")) %>%
  mutate(countryID = as.integer(countryID))

country_summary2 <- estimates_keras_tbl %>%
  select(c("country", "subregion", "Situation")) %>%
  mutate_at(vars("Situation"), .funs = factor) %>%
  data.table() %>%
  mltools::one_hot(cols=c("Situation")) %>%
  dplyr::group_by(country) %>%
  summarize(count =  n(), 
            n_victimized =  sum(Situation_Victimized),
            n_escaped = sum(Situation_Escaped),
            n_ideal_citizen = sum(`Situation_Ideal_Citizen`),
            n_incarcerated =  sum(Situation_Incarcerated))  %>%
  mutate(p_victimized = round(100*n_victimized/count), 
         p_escaped = round(100*n_escaped/count),
         p_ideal_citizen = round(100*n_ideal_citizen/count),
         p_incarcerated = round(100*n_incarcerated/count)) %>%
  mutate(Specificity = n_ideal_citizen/(n_ideal_citizen+n_victimized),
         Sensitivity = n_incarcerated/(n_incarcerated+n_escaped)) %>%
  select(-c("count", "n_victimized", "n_escaped", "n_ideal_citizen", "n_incarcerated")) %>%
  mutate(p_correct = p_incarcerated + p_ideal_citizen) %>%
  left_join(country_values, by=c("country"="country")) %>%
  mutate(country = substring(country, 6))

pointsToLabel <- c("Israel","Bangladesh","Iraq","Tanzania","Uganda",
                   "France", "Canada","Australia","NewZealand",
                   "USA", "Germany", "UnitedKingdom",
                   "NewZealand")

ggplot(country_summary2,aes(y=Specificity,x=Sensitivity,color=region)) +
  geom_point(size=3,shape=1,stroke = 1, alpha=0.7) +
  #geom_smooth(aes(group=1),method ='lm',formula = y~x,se=FALSE,color='red') +
  geom_text_repel(aes(label = country), color = "gray20",
                  vjust = -1, nudge_y = 0.1,
                  size=4,
                  data = subset(country_summary2, country %in% pointsToLabel)) +
  theme_bw() +
  scale_y_continuous(name = "Specificity") +
  scale_x_continuous(name = "Sensitivity") +
  ggtitle("Predicting Values") +
  theme_economist_white()

ggsave(wmc_image, width = 9, height = 6, units =  "in")

mapped_data <- joinCountryData2Map(country_summary2, joinCode = "ISO3", 
                                   nameJoinColumn = "ISO3")  %>%
  subset(continent != "Antarctica")
par(mai=c(0,0,0.2,0),xaxs="i",yaxs="i")
png(FTA_image,width=9,height=5,units="in", res=300)
FTA_map <- mapCountryData(mapToPlot =         mapped_data
                          , nameColumnToPlot =  "p_correct"
                          , numCats =           6
                          , xlim =              NA
                          , ylim =              NA
                          , mapRegion =         "world"
                          , catMethod =         "pretty"
                          , colourPalette =     c("#ff0000","#ff6600","#ffcc00","#cbff00","#65ff00","#00ff00")
                          , addLegend =         FALSE
                          , borderCol =         'grey'
                          , mapTitle =          'Percentage of unseen cases correctly classified'
                          , oceanCol =          NA
                          , aspect =            1
                          , missingCountryCol = "light grey"
                          , add =               FALSE
                          , nameColumnToHatch = ""
                          , lwd =               0.7)

#FTA_map$legendText <- c("CAN", "MEX", "USA")
do.call(addMapLegendBoxes, c(FTA_map,x="bottomleft",title = "Accuracy",horiz=FALSE))
dev.off()

#
pred_cor <- estimates_keras_tbl %>% filter(correct == "correct")
pred_wrong <- estimates_keras_tbl %>% filter(correct == "wrong")

#
#  Choose cases to explain one case from each subregion
#
subregion_columns <- dplyr::select(data.frame(test_input_matrix), matches("^SR_"))
subregion_names <- names(subregion_columns)

options(yardstick.event_first = FALSE)
# Confusion Table
print("Confusion for Keras")
estimates_keras_tbl %>% conf_mat(keras_truth, keras_estimate)

# Accuracy
print("Accuracy for Keras")
estimates_keras_tbl %>% metrics(keras_truth, keras_estimate)

# AUC
print("AUC for Keras")
estimates_keras_tbl %>% roc_auc(keras_truth, keras_class_prob)

# Precision for keras
print("Precision/Recall for Keras")
pre_rec <- tibble(
  keras_precision = estimates_keras_tbl %>% yardstick::precision(keras_truth, keras_estimate),
  keras_recall    = estimates_keras_tbl %>% yardstick::recall(keras_truth, keras_estimate)
)
print(pre_rec)

#
#F1 score
#

print("F1 score for Keras")
estimates_keras_tbl %>% f_meas(keras_truth, keras_estimate, beta = 1)

#
# Lime
#

#
# There in an incompatity between packages. The predictions generate Error. So this is a workaround
# suggested by https://github.com/thomasp85/lime/issues/139
#

class(model_keras)
model_type.keras.engine.sequential.Sequential <- function(x, ...) {
  "classification"}

predict_model.keras.engine.sequential.Sequential <- function (x, newdata, type, ...) {
  pred <- predict_proba (object = x, x = as.matrix(newdata))
  data.frame (High = pred, Low = 1 - pred) }

predict_model (x       = model_keras,
               newdata = test_input_matrix,
               type    = 'raw') %>%
  tibble::as_tibble()

explainer_keras <- lime::lime (
  x              = as_tibble(train_input_matrix),
  model          = model_keras,
  bin_continuous = FALSE)

#
# Explain the 1 from each subregion  predictions
#
explanation_keras_subregion <- lime::explain (
  x = test_case_each_subregion,
  explainer    = explainer_keras,
  n_labels     = 1, # explaining a `single class`(binary)
  n_features   = 5)

plot_features (explanation_keras_subregion, ncol = 2) +
  labs (title = paste("World Values (", regress_var, "): Feature Importance Visualization  in Subregions"),
        subtitle = "Hold Out (Test) Set: Correct Predictions") + theme_grey(base_size = 12)
ggsave(fivsr_image, width = 13, height = 14, units =  "in")


# #
# # Explain the 4 correct singapore predictions
# #
# explanation_keras_singapore <- lime::explain (
#   x = test_data_singapore,
#   explainer    = explainer_keras,
#   n_labels     = 1, # explaining a `single class`(binary)
#   n_features   = 8)
# 
# plot_features (explanation_keras_singapore, ncol = 4) +
#   labs (title = paste("World Values (", regress_var, "): Feature Importance Visualization  in Singapore"),
#         subtitle = "Hold Out (Test) Set: Correct Predictions")
# ggsave(fivs_image, width = 9, height = 7, units =  "in")
# 


# pic_label <- paste("World Values (", regress_var, "): Feature Importance Heatmap")
# plot_explanations (explanation_keras_cor) +
#   labs (title = pic_label,
#         subtitle = "Hold Out (Test) Set")
# ggsave(fih_image, width = 6, height = 5, units =  "in")

corrr_analysis <- as_tibble(train_input_matrix) %>%
  dplyr::mutate (regress_var = as.vector(train_output_matrix)) %>%
  correlate () %>%
  focus (regress_var) %>%
  rename (feature = rowname) %>%
  arrange (abs(regress_var)) %>%
  mutate (feature = as_factor(feature)) %>%
  filter(regress_var < -0.09 | regress_var > 0.09)

corrr_analysis %>%
  ggplot (aes (x = regress_var, y = fct_reorder(feature, desc(regress_var)))) +
  geom_point () +
  # Positive Correlations - Contribute to Polarity--------------------------------------------
  geom_segment (aes(xend = 0, yend = feature),
              color = palette_light()[[2]],
              data = corrr_analysis %>% filter(regress_var > 0)) +
  geom_point (color = palette_light()[[2]],
              data = corrr_analysis %>% filter(regress_var > 0)) +
  # Negative Correlations - Prevent Polarity--------------------------------------------------
  geom_segment (aes(xend = 0, yend = feature),
              color = palette_light()[[1]],
              data = corrr_analysis %>% filter(regress_var < 0)) +
  geom_point (color = palette_light()[[1]],
              data = corrr_analysis %>% filter(regress_var < 0)) +
  # Vertical lines-------------------------------------------------------------------------
  geom_vline (xintercept = 0, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline (xintercept = -0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  geom_vline (xintercept = 0.25, color = palette_light()[[5]], size = 1, linetype = 2) +
  # Aesthetics-----------------------------------------------------------------------------
  labs (title = "Classification Correlation Analysis",
        subtitle = "Positive Correlations vs. Negative Correlations",
        y = "Feature Importance") +
  theme_bw () 

ggsave(fi_image, width = 10, height = 49, units =  "in")


#
# Dalex comparisons of features
#

all_names <- colnames(train_input_matrix)
r1 <- grep("^s002", all_names, value = TRUE)
r2 <- grep("^s003", all_names, value = TRUE)
r3 <- grep("^sub_region", all_names, value = TRUE)
focus_on_vars <- all_names %>% setdiff(c(r1,r2,r3))

explainer_keras_dalex <- DALEX::explain(model_keras,
                                        data = as.matrix(train_input_matrix),
                                        y = as.numeric(train_output_matrix),
                                        variables = focus_on_vars,
                                        label = "Seen Model",
                                        predict_function = function(m,x) keras_predict(m, x, verbose=2))

#model performance
mp_keras_dalex <- model_performance(explainer_keras_dalex)
ggplot(mp_keras_dalex, aes(observed, diff)) + geom_point(color="red",size=5, alpha=0.007) +
  xlab("Observed") + ylab("Predicted - Observed") +
  ggtitle("Diagnostic plot for the Keras model") + theme_mi2()
ggsave(mck_image, width = 6, height = 8, units =  "in")

mp_lm_dalex <- model_performance(explainer_lm_dalex)
ggplot(mp_lm_dalex, aes(observed, diff)) + geom_point(color="red",size=5, alpha=0.007) +
  xlab("Observed") + ylab("Predicted - Observed") +
  ggtitle("Diagnostic plot for the GLM") + theme_mi2()
ggsave(mcl_image, width = 6, height = 8, units =  "in")

#
# which has largest residual
#
print("Largest Residual")
new_obs <- test_input_matrix[which.min(mp_keras_dalex$diff), ]
print(new_obs)

# use type = "raw" for raw plotting
vi_keras_dalex <- feature_importance(explainer_keras_dalex,
                                      loss_function = loss_root_mean_square)

p_seen <- plot(vi_keras_dalex,  max_vars = 20)
ggsave(vic_image, width = 6, height = 8, units =  "in", plot = p_seen)

head(vi_keras_dalex[order(-vi_keras_dalex$dropout_loss),],100) %>%
  write.csv(world_file)

#Reviewers comment
explainer_keras_dalex_unseen <- DALEX::explain(model_keras,
                                        data = as.matrix(test_input_matrix),
                                        y = as.numeric(test_output_matrix),
                                        variables = focus_on_vars,
                                        label = "Uneen Model",
                                        predict_function = function(m,x) keras_predict(m, x, verbose=2))

vi_keras_dalex_unseen <- feature_importance(explainer_keras_dalex_unseen,
                                     loss_function = loss_root_mean_square)

p_unseen <- plot(vi_keras_dalex_unseen,  max_vars = 20)
p <- grid.arrange(p_seen, p_unseen, ncol = 2)
ggsave(vic_image, width = 10, height = 10, units =  "in", dpi=300, plot=p)

#https://www.r-bloggers.com/explaining-black-box-machine-learning-models-code-part-1-tabular-data-caret-iml/

sv_keras_v1 <- model_profile(explainer_keras_dalex, variables =  c("f194", "d067", "b017_1"), N=1000)

options(digits = 3)

plot(sv_keras_v1)
ggsave(pdp_v1_image, width = 8, height = 4, units =  "in")

#Do explanation across waves
wave_columns <- dplyr::select(data.frame(train_input_matrix), matches("s002_"))
wave_names <- names(wave_columns)
wave_data <- cbind(as_tibble(train_input_matrix), as_tibble(train_output_matrix))
wave_predictors <- data.frame(
  variable=character(),
  dropout_loss=double(),
  label=character()
)
for (wave in wave_names) {
  wave_frame <- wave_data %>%
    filter_(interp(~ var == 1, var = as.name(wave)))
  print(paste("working on", wave, "Got Observations", nrow(wave_frame)))
  wave_input_matrix <- select(wave_frame, -c(regress_var)) %>% as.matrix()
  wave_output_matrix <- select(wave_frame, c(regress_var)) %>% as.matrix() %>% as.numeric()
  wave_explainer <- DALEX::explain(model_keras,
                                   data = wave_input_matrix,
                                   y = wave_output_matrix,
                                   label = wave,
                                   predict_function = function(m,x) keras_predict(m, x, verbose=2))
  vi_wave <- feature_importance(wave_explainer,
                                 loss_function = loss_root_mean_square)
  wave_predictors <- wave_predictors %>% rbind(head(vi_wave[order(-vi_wave$dropout_loss),],100))
  fname <- paste(wave, "png", sep = ".")
  wave_image  <- paste(basedatadir, fname, sep='/')
  plot(vi_wave, max_vars = 50)
  ggsave(wave_image, width = 6, height = 7, units =  "in")
}
write.csv(wave_predictors, wave_file)

#Do explanation subregionwide

subregion_columns <- dplyr::select(data.frame(train_input_matrix), matches("^sub_region_code"))
subregion_names <- names(subregion_columns)
subregion_data <- cbind(as_tibble(train_input_matrix), as_tibble(train_output_matrix))
subregion_predictors <- data.frame(
  variable=character(),
  dropout_loss=double(),
  label=character()
)
for (subregion in subregion_names) {
  subregion_frame <- subregion_data %>%
    filter_(interp(~ var == 1, var = as.name(subregion)))
  print(paste("working on", subregion, "Got Observations", nrow(subregion_frame)))
  subregion_input_matrix <- select(subregion_frame, -c(regress_var)) %>% as.matrix()
  subregion_output_matrix <- select(subregion_frame, c(regress_var)) %>% as.matrix() %>% as.numeric()
  subregion_explainer <- DALEX::explain(model_keras,
                                        data = subregion_input_matrix,
                                        y = subregion_output_matrix,
                                        label = subregion,
                                        predict_function = function(m,x) keras_predict(m, x, verbose=2))
  vi_subregion <- feature_importance(subregion_explainer,
                                      loss_function = loss_root_mean_square)
  subregion_predictors <- subregion_predictors %>% rbind(head(vi_subregion[order(-vi_subregion$dropout_loss),],100))
  fname <- paste(subregion, "png", sep = ".")
  subregion_image  <- paste(basedatadir, fname, sep='/')
  plot(vi_subregion, max_vars = 50)
  ggsave(subregion_image, width = 6, height = 7, units =  "in")
}
write.csv(subregion_predictors, subregion_file)

#Do explanation across countries
country_columns <- dplyr::select(data.frame(train_input_matrix), matches("^s003"))
country_names <- names(country_columns)
country_data <- cbind(as_tibble(train_input_matrix), as_tibble(train_output_matrix))
country_predictors <- data.frame(
  variable=character(),
  dropout_loss=double(),
  label=character()
)

for (country in country_names) {
  country_frame <- country_data %>%
    filter_(interp(~ var == 1, var = as.name(country)))
  print(paste("working on", country, "Got Observations", nrow(country_frame)))
  country_input_matrix <- select(country_frame, -c(regress_var)) %>% as.matrix()
  country_output_matrix <- select(country_frame, c(regress_var)) %>% as.matrix() %>% as.numeric()
  country_explainer <- DALEX::explain(model_keras,
                                   data = country_input_matrix,
                                   y = country_output_matrix,
                                   label = country,
                                   predict_function = function(m,x) keras_predict(m, x, verbose=2))
  vi_country <- feature_importance(country_explainer,
                                 loss_function = loss_root_mean_square)

  country_predictors <- country_predictors %>% rbind(head(vi_country[order(-vi_country$dropout_loss),],100))
  fname <- paste(country, "png", sep = ".")
  country_image  <- paste(basedatadir, fname, sep='/')
  plot(vi_country, max_vars = 50)
  ggsave(country_image, width = 6, height = 7, units =  "in")
}
write.csv(country_predictors, country_file)

#pairwise interaction

fl <- flashlight(model = model_keras,
                 data = as.data.frame(train_input_matrix),
                 #y = "care_environment",
                 label = "Binary CrossEntropy",
                 predict_function = function(m,x) keras_predict(m,
                                                                as.matrix(x), 
                                                                verbose=2))

pattern <- "^s002|^s003|^sub_region|^region_code"
to_remove <- grep(pattern, colnames(train_input_matrix), value=TRUE)
x <- setdiff(colnames(train_input_matrix), to_remove)

st <- light_interaction(fl, v = x)
plot(st)
ggsave(absolute_interaction, width = 6, height = 40, units =  "in", dpi=150)
#st_pair <- light_interaction(fl, v = most_important(st, 10), pairwise = TRUE)
st_pair <- light_interaction(fl, v = most_important(st, 200), pairwise = TRUE)
plot(st_pair)
ggsave(pairwise_interaction, width = 6, height = 30, units =  "in")


