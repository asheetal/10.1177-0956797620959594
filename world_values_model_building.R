#!/usr/bin/Rscript 
library(keras, quietly = TRUE)
use_implementation("keras")
use_backend("plaidml")
library(testit)
assert("Using Plaidml", k_backend() == "plaidml")
library(dplyr, quietly = TRUE)
library(varhandle, quietly = TRUE)
library(data.table, quietly = TRUE)
library(mltools, quietly = TRUE)
library(caTools, quietly = TRUE)
library(tibble, quietly = TRUE)
library(filenamer, quietly = TRUE)
library(readxl, quietly = TRUE)
library(stringi)
library(caret)
library(jsonlite, quietly = TRUE)
library(httr)
library(RCurl)
INT_NA <- -9

base_url <- "https://hyperparam.cloud/api/sheetal/1283"
get_url <- paste(base_url, "get", sep="/")
res_url <- paste(base_url, "res", sep='/')
put_url <- paste(base_url, "put", sep='/')

token <- paste("Bearer", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJpdnl0aWVzIiwibmFtZSI6InNoZWV0YWwifQ.WM-RQHS6agJmJtKRnLqm3u5dcIXNPyp0yoioYcXfWUY")
options(digits = 16)


hyperparameter_flow = FALSE
load("/research/dataset/world_values/unethical/hyperparameter_environment_2020-03-17.RData")

print("Loaded image")

if (Sys.getenv("HYPER") == 1) {
  hyperparameter_flow <- TRUE
  print("Hyperparameter run")
}

#Hyperparameter
FLAGS <- flags(
  flag_numeric("R_FLAGS_MY_UNITS1", 900),
  flag_numeric("R_FLAGS_MY_UNITS2", 479),
  flag_numeric("R_FLAGS_MY_UNITS3", 225),
  flag_numeric("R_FLAGS_MY_UNITS4", 46),
  flag_numeric("R_FLAGS_DROPOUT1", 0.2101),
  flag_numeric("R_FLAGS_DROPOUT2", 0.166),
  flag_numeric("R_FLAGS_DROPOUT3", 0.6732),
  flag_numeric("R_FLAGS_DROPOUT4", 0.1455),
  flag_string("R_FLAGS_MY_OPTIMIZER", "adam"),
  flag_numeric("R_FLAGS_MY_BATCHSIZE", 64),
  flag_numeric("R_FLAGS_MY_LR", 460)
)

if (hyperparameter_flow) {
  print(Sys.getenv("PLAIDML_DEVICE_IDS"))
  req <- httr::GET(get_url, httr::add_headers(Authorization = token))
  stop_for_status(req)
  proposition_json <- content(req)
  FLAGS <- as.data.frame(proposition_json$prop.points)
  FLAGS$R_FLAGS_MY_OPTIMIZER  <- "adam"
  print(FLAGS)
}

if (FALSE) {
#change these only
basedir <- "/research/dataset/world_values"
regress_var <- "unethical"
all_variables <- read_excel("/research/dataset/world_values/unethical/Variables_to_keep_2019_11_28.xlsx") #without s003
all_variables <- read_excel("/research/dataset/world_values/unethical/Variables_to_keep_2020-03-17.xlsx") #with s003
trainfile <- "/research/dataset/world_values/unethical/imputed_train_2019-10-22.rds"


#this is a very time consuming program. Must put timestamp to output files
basedatadir <- paste(basedir,regress_var,sep='/')
options(filenamer.timestamp=1)

all_dependant_vars <- c("f114", "f115", "f116", "f117")
my_dependant_vars <- c("claimbenefits", "unethical", "avoidingfare", "cheatingtax", "bribery")
drop_non_regress_vars <- setdiff(my_dependant_vars, regress_var)

#some variables are one hot coded but must be dropped first
one_hot_drop <- all_variables %>%
  filter((Dummy ==  1) & (Delete_Before_Dishonesty ==  1)) %>%
  pull(1) %>%
  paste0("_", sep="")

one_hot_drop <- paste("^", one_hot_drop, sep="")

#secondly drop rest of drop variables
drop_more_vars <- all_variables %>%
  filter(Delete_Before_Dishonesty ==  1) %>%
  pull(1)

df7.temp <- readRDS(trainfile)

vGrep <- Vectorize(grep, vectorize.args = "pattern", SIMPLIFY = FALSE)
grep_list <- vGrep(one_hot_drop, names(df7.temp), value=TRUE)

final_drop_list <- union(drop_more_vars, unlist(grep_list, use.names = FALSE)) %>%
  union(drop_non_regress_vars)

final_keep_vars <- setdiff(names(df7.temp), final_drop_list)

training_image <- filename("training",
                           path=basedatadir,
                           tag=NULL,
                           ext="png",
                           subdir=FALSE) %>%
  as.character() %>%
  print()

model_file <- filename("model",
                       path=basedatadir,
                       tag=NULL,
                       ext="hd5",
                       subdir=FALSE) %>%
  as.character() %>%
  print()

hyperparamete_env <- filename("hyperparameter_environment",
                              path=basedatadir,
                              tag=NULL,
                              ext="RData",
                              subdir=FALSE) %>%
  as.character() %>%
  print()


my_shuffle <- function (df=NULL) {
  rows <- sample(nrow(df))
  return(df[rows, ])
}

create_label <- function(df=NULL){
  ret_df <- df %>%
  mutate(claimbenefits = case_when(
    f114 == 1 ~ 0,
    f114 > 1 ~ 1,
    TRUE ~ INT_NA
  )) %>%
    mutate(avoidingfare = case_when(
      f115 == 1 ~ 0,
      f115 > 1 ~ 1,
      TRUE ~ INT_NA
    )) %>%
    mutate(cheatingtax = case_when(
      f116 == 1 ~ 0,
      f116 > 1 ~ 1,
      TRUE ~ INT_NA
    )) %>%
    mutate(bribery = case_when(
      f117 == 1 ~ 0,
      f117 > 1 ~ 1,
      TRUE ~ INT_NA
    )) %>%
    mutate(unethical = case_when(
      ((claimbenefits == 1) | (avoidingfare == 1) | (cheatingtax == 1) | (bribery == 1)) ~ 1,
      TRUE ~ 0
    ))
  return(ret_df)
}


df7.train <-  df7.temp %>%
  create_label() %>%
  select(c(final_keep_vars, regress_var)) %>%
  my_shuffle()

df7.temp <- NULL
  
rownames(df7.train) <- NULL

preProcessModel <- dplyr::select(df7.train, -c(regress_var)) %>%
  preProcess(method = "range")

train_input_matrix <- predict(preProcessModel, dplyr::select(df7.train, -c(regress_var))) %>%
  as.matrix()

train_output_matrix <- as.matrix(dplyr::select(df7.train, c(regress_var)))

number_output_vars <- ncol(train_output_matrix)
number_input_vars <- ncol(train_input_matrix)
#
# flush old vars to reduce memory
#
df7.train <- NULL
save.image(file=hyperparamete_env)
} 

my_seed <- 101
set.seed(my_seed)

model_keras <- NULL
model_keras <- keras_model_sequential()
my_early_stopping_patience <- 10
my_lr_patience <- 5
epochs <- 200

#
# let hyperparameter control optimizer choice and learning rate
#

my_lr <- FLAGS$R_FLAGS_MY_LR * 10^(-6)
switch (FLAGS$R_FLAGS_MY_OPTIMIZER,
        "adam" = my_optimizer  <- optimizer_adam(lr=my_lr)
        ,"adadelta" = my_optimizer  <- optimizer_adadelta(lr=my_lr)
        ,"adagrad" = my_optimizer  <- optimizer_adagrad(lr=my_lr)
        ,"rmsprop" = my_optimizer <- optimizer_rmsprop(lr=my_lr)
        ,"nadam" = my_optimizer <- optimizer_nadam(lr=my_lr)
        ,"sgd" = my_optimizer <- optimizer_sgd(lr=my_lr)
)

model_keras <- model_keras %>%
  layer_dense(input_shape = number_input_vars,
              units = FLAGS$R_FLAGS_MY_UNITS1,
              kernel_initializer = "ones") %>%
  layer_activation("relu")  %>%
  layer_gaussian_dropout(rate = FLAGS$R_FLAGS_DROPOUT1) %>%
  layer_batch_normalization() %>%
  layer_dense(units = FLAGS$R_FLAGS_MY_UNITS2,
              kernel_initializer = "ones") %>%
  layer_activation("relu")  %>%
  layer_gaussian_dropout(rate = FLAGS$R_FLAGS_DROPOUT2) %>%
  layer_batch_normalization() %>%
  layer_dense(units = FLAGS$R_FLAGS_MY_UNITS3,
              kernel_initializer = "ones") %>%
  layer_activation("relu")  %>%
  layer_gaussian_dropout(rate = FLAGS$R_FLAGS_DROPOUT3) %>%
  layer_batch_normalization() %>%
  layer_dense(units = FLAGS$R_FLAGS_MY_UNITS4,
              kernel_initializer = "ones") %>%
  layer_activation("relu")  %>%
  layer_gaussian_dropout(rate = FLAGS$R_FLAGS_DROPOUT4) %>%
  layer_batch_normalization() %>%
  layer_dense(units = number_output_vars, #
              kernel_initializer = "zeros") %>%
  layer_activation("sigmoid")  %>%
  compile(loss = "binary_crossentropy",
          optimizer = my_optimizer,
          metrics = c("accuracy")) #accuracy, mae

if (hyperparameter_flow) {
  callbacks = list(
   callback_early_stopping(monitor="val_loss", patience = my_early_stopping_patience)
    ,callback_reduce_lr_on_plateau(monitor="val_loss", patience=my_lr_patience, factor = 0.5)
  )
} else {
  callbacks = list(
    callback_early_stopping(monitor="val_loss", patience = my_early_stopping_patience)
    ,callback_reduce_lr_on_plateau(monitor="val_loss", patience=my_lr_patience, factor = 0.5)
    ,callback_model_checkpoint(
      filepath = model_file,
      save_best_only = TRUE,
      period = 1,
      verbose = 0
    )
  )
}
  
fit_verbose  <- ifelse(hyperparameter_flow, 0, 1)

history <- fit(
  object = model_keras,
  x = train_input_matrix, 
  y = train_output_matrix, 
  epochs = epochs, 
  batch_size = as.integer(paste(FLAGS$R_FLAGS_MY_BATCHSIZE)),
  verbose = fit_verbose
  ,callbacks
  ,validation_split = 0.2
)

if(hyperparameter_flow)  {
  print(FLAGS)
  print(history)
  y <- min(history$metrics$val_loss)
  print(paste("Smallest Validation Loss: ", y))
  prop.points <- proposition_json$prop.points
  send_list <- unlist(append(prop.points,list(y=y)), recursive = F)
  #print(send_list)
  req <- httr::POST(put_url, body=send_list, httr::add_headers(Authorization = token), encode = "json")
  stop_for_status(req)
  content(req)
  
  req <- httr::GET(res_url, httr::add_headers(Authorization = token))
  stop_for_status(req)
  res <- do.call(rbind.data.frame, content(req))
  print(res)
  gc(reset = TRUE)
} else {
  #print the training graph
  png(filename=training_image, height = 4, width = 6, res = 300, units = "in")
  print(plot(history))
  dev.off()
}

