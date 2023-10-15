#!/usr/bin/Rscript 
library(Amelia)
library(missRanger)
library(dplyr)
library(parallel)
library(bcv)
library(varhandle)
library(doParallel)
library(data.table)
library(mltools)
library(haven)
library(caTools)
library(filenamer)
library(readxl)
library(testit)
library(janitor)
library(Hmisc)

input_file <- "/research/dataset/world_values/WorldValuesSurvey20190429.dta"
basedatadir <- "/research/dataset/world_values/unethical"

all_variables <- read_excel("/research/dataset/world_values/unethical/Variables_to_keep_2020-03-17.xlsx")

drop_these_vars <- all_variables %>%
  filter(Delete_Before_Imputing==1) %>%
  select("VARIABLE") %>%
  pull(1)

one_hot_list <- all_variables %>%
  filter(Dummy==1) %>%
  select("VARIABLE") %>%
  pull(1)

#this is a very time consuming program. Must put timestamp to output files
options(filenamer.timestamp=1)
imputed_filename <- filename("imputed_train", 
                          path=basedatadir, 
                          tag=NULL, 
                          ext="rds", 
                          subdir=FALSE) %>%
                    as.character() %>%
                    print()
unseen_filename <- filename("unseen_test", 
                            path=basedatadir, 
                            tag=NULL, 
                            ext="rds", 
                            subdir=FALSE) %>%
                  as.character() %>%
                  print()

df2 <- read_dta(input_file) %>%
  filter((f114 > 0) | (f115 > 0) | (f116 > 0) | (f117 > 0))

set.seed(101)

#all negatives are NA
df2[df2 < 0] <- NA
#drop all columns that have only NA
df2<-Filter(function(x) !all(is.na(x)), df2)

#
#what are the columns that do not appear in the code file?
#
my_column_names <- names(df2)
coding_mismatch <-  setdiff(my_column_names, all_variables$VARIABLE)
assert("Difference between Excel names and dataframe names is zero", length(coding_mismatch)==0)


#
# there is a bug in dataset where the master key code talks about some variables that do not exists in actual dataset.
# Cannot drop something that does not exist
#
keep_column_names <- setdiff(my_column_names, drop_these_vars)

df4 <- df2 %>% 
  dplyr::select(keep_column_names)

#how many missing values are there?
print("how many missing values are there?")
mean(is.na(df4))

#check all columns that have many levels 
sort(sapply(df4, function(col) length(unique(col))))

#some one hot list items are already dropped out
missing_one_hot_list <- setdiff(one_hot_list, names(df4))
final_one_hot_list <- setdiff(one_hot_list, missing_one_hot_list)

#change some vars to one hot encoding
df5 <- df4 %>% 
  mutate_at(vars(final_one_hot_list), .funs = factor)

df5 <- mltools::one_hot(as.data.table(df5), cols = c(final_one_hot_list))
sort(sapply(df5, function(col) length(unique(col))),  decreasing=TRUE)

#some columns are still text here
df6 <- sapply(df5, as.numeric) %>% as.data.frame()
#
#SPEARMAN CORR with DV
#
create_label <- function(df=NULL){
  ret_df <- df %>%
    dplyr::mutate(claimbenefits = case_when(
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

df4_corr <- df6 %>% create_label() %>% select(-c("f114", "f115", "f116", "f117")) %>%
  select(-c("claimbenefits", "avoidingfare", "cheatingtax", "bribery"))
df4_corr[df4_corr==INT_NA] <- NA
mean(is.na(df4_corr))
corr_list <- Hmisc::rcorr(as.matrix(df4_corr), type=c("pearson"))
r <- corr_list[["r"]] %>% as.data.frame() %>% select("unethical")
n <- corr_list[["n"]] %>% as.data.frame() %>% select("unethical")
p <- corr_list[["P"]] %>% as.data.frame() %>% select("unethical")
corr_df <- data.frame(n, r, p)
names(corr_df) <- c("N", "R", "P")
options(filenamer.timestamp=1)
spearman_file <- filename("pearson_pairwise_corr",
                          path=basedatadir,
                          tag=NULL,
                          ext="csv",
                          subdir=FALSE) %>%
  as.character() %>%
  print()
write.csv(corr_df, spearman_file)

#test train split
data1 <- sample.split(df6,SplitRatio = 0.1)
#subsetting into Train data
df6.train <- subset(df6,data1==FALSE)
#subsetting into Test data
df6.test <- subset(df6,data1==TRUE)
saveRDS(df6.test, unseen_filename)

df7 <- df6.train %>%
  missRanger(verbose = 2, 
             num.trees=100, 
             maxiter=15,
             respect.unordered.factors=TRUE,
             splitrule = "extratrees")

saveRDS(df7, imputed_filename)
