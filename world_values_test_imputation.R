#!/usr/bin/Rscript 
library(dplyr) ##be careful with explain function
library(scales)
library(filenamer)
library(readxl, quietly = TRUE)
library(robustbase)
library(data.table)
library(missRanger)

#change only these
basedir <- "/research/dataset/world_values"
regress_var <- "unethical"
basedatadir <- paste(basedir, regress_var, sep='/')

#load the test and imputed training files
trainfile  <- "/research/dataset/world_values/unethical/imputed_train_2019-10-22.rds"
testfile  <- "/research/dataset/world_values/unethical/unseen_test_2019-10-22.rds"

options(filenamer.timestamp=1)
imputed_unseen_filename <- filename("imputed_unseen",
                           path=basedatadir,
                           tag=NULL,
                           ext="rds",
                           subdir=FALSE) %>%
  as.character() %>%
  print()

#perform same operations as above on the unseen data
df7.train <- readRDS(trainfile)
df7.test <- readRDS(testfile)

#how  many missing values are there
mean(is.na(df7.test))

combined_df  <- rbind(df7.train, df7.test)
unseen_index <- (nrow(df7.train)+1):nrow(combined_df)
mean(is.na(combined_df))

combined_df.imputed <- combined_df %>%
              missRanger(verbose = 2, 
                         num.trees=100, 
                         maxiter=15,
                         respect.unordered.factors=TRUE,
                         splitrule = "extratrees")

unseen_imputed <- combined_df.imputed[unseen_index,]
saveRDS(unseen_imputed, imputed_unseen_filename)
