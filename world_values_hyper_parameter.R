#!/usr/bin/Rscript 
library(tfruns)

runs <- tuning_run("~/Work/Experiments/Krishna/world_values_model_building.R", 
                   runs_dir = "world_values_hyperparameter_runs",
                   sample = 0.01,
                   flags = list(
                                my_dropout1 = c(0.3, 0.4, 0.5, 0.6),
                                my_dropout2 = c(0.3, 0.4, 0.5, 0.6),
                                my_dropout3 = c(0.3, 0.4, 0.5, 0.6),
                                my_dropout4 = c(0.3, 0.4, 0.5, 0.6),
                                my_optimizer = c("sgd", "rmsprop","adam", "adamax", "adadelta"),
                                my_batchsize = c(100, 200, 250)
))

# find the best evaluation accuracy
runs[order(runs$eval_acc, decreasing = TRUE), ]
