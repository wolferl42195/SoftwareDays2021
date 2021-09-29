# load libs
library("pacman");
p_load(xts, zoo, TTR, tidyverse, readr, dplyr, lubridate)
p_load(quantmod)
p_load(xgboost)
p_load(PerformanceAnalytics)
library(ROCR)

# setup
constant <- 42
EXPORT_DIR = "export/Softwareday/"
DATA_DIR   = "data/"

symbol = "EBS.VI"

start <- as.Date("2018-06-01")
end   <- as.Date("2019-03-31")

nDays       <- 14
splitfactor <- 0.75

eta         <- 0.0001
max_depth   <- 7
nrounds     <- 20
eval_metric <- "auc"

# load
#load market data
filename <- paste(DATA_DIR, symbol, ".csv", sep = '')
quotes  = read.zoo(file=filename,header=TRUE,  sep = ",") %>% as.xts

quotes <- quotes["20180831/20190331"]

# prepare marketdata
closePrices <- Cl(quotes)
volumes <- Vo(quotes)

# charting only
#doChart(closePrices)
candleChart(closePrices, theme='white') 
addTA(closePrices) 
addSMA(ndays) # Add Simple Moving Average

# Returns
set.seed(constant)
ret  <- Return.calculate(closePrices)

# combine technical indicators
strats = merge(closePrices, ret) %>% na.omit
colnames(strats) <- c("Price","return")

quote_data = as_tibble(coredata(strats))
quote_data = quote_data %>% mutate(qDate = index(strats))

# merge marketdata/technical indicators with news features

data = quote_data

# labeling
set.seed(constant)
return_lead      = lead(data$return, n=1)             # shift by 1 day
sign_return_lead = sign(return_lead)
lead_target      = if_else(sign_return_lead==1, 1, 0)

data <- data %>% mutate(return_lead, sign_return_lead, lead_target)
data <- data %>% na.omit


# Added to analyse features
# Understanding the dataset using descriptive statistics
set.seed(constant)
print(head(data),5)
y = data$lead_target
cbind(freq=table(y), percentage=prop.table(table(y))*100)
summary(data)

# split in train and test data
set.seed(constant)
total_n      =  data %>% nrow()
length_train =  floor(total_n * splitfactor)
length_test  =  total_n - length_train

train_data = data %>% top_n( -length_train, qDate)
test_data  = data %>% top_n(  length_test,  qDate)

var_ex = c("qDate", "Price", "return_lead", "sign_return_lead", "lead_target")
X_train <- train_data %>% select(-one_of(var_ex)) %>% as.matrix()
Y_train <- train_data$lead_target

X_test  <- test_data %>% select(-one_of(var_ex)) %>% as.matrix()
Y_test  <- test_data$lead_target

# xgboost
set.seed(constant)
xgb.train.data <- xgb.DMatrix(data = X_train, label = Y_train)
xgb.test.data  <- xgb.DMatrix(data = X_test , label = Y_test )

watchlist      <- list(train = xgb.train.data, val = xgb.test.data)

tree.params = list(
  booster = "gbtree", 
  eta              = eta, 
  max_depth        = max_depth, 
  min_child_weight = 1, 
  subsample        = 0.6, 
  colsample_bytree = 1, 
  gamma = 0.0, 
  objective = "binary:logistic")

xgb.model.tree = xgb.train(
  data          = xgb.train.data, 
  weight = NULL,  
  watchlist = watchlist,
  params = tree.params, 
  nrounds       = nrounds, 
  verbose = 2, 
  print_every_n = 10L, 
  eval_metric   = eval_metric, 
  maximize = TRUE,
  early_stopping_rounds = 10) 

set.seed(constant)
train_preds = predict(xgb.model.tree, X_train)
test_preds  = predict(xgb.model.tree, X_test)

print(head(train_preds))
print(head(test_preds))

# cv cross validation
xgb_params    = list(
  objective   = "binary:logistic",                                    
  eta         = eta,                                                  
  max.depth   = max_depth,                                                    
  eval_metric = eval_metric)                                                

xgb_cv = xgb.cv(
  params        = xgb_params,
  data          = X_train,
  label         = Y_train,
  nrounds       = nrounds, 
  nfold         = 3,                                
  print_every_n = 1, 
  early_stopping_rounds = 10)

# end cv cross validation

# confusionMatrix
confMatrixTest  <- caret::confusionMatrix(Y_test  %>% as.factor, ifelse(test_preds  > 0.5, 1, 0) %>% as.factor)
confMatrixTrain <- caret::confusionMatrix(Y_train %>% as.factor, ifelse(train_preds > 0.5, 1, 0) %>% as.factor)

print(confMatrixTest)
print(confMatrixTrain)

xgb.pred <- prediction(test_preds, Y_test)
xgb.perf <- performance(xgb.pred, "tpr", "fpr")

plot(xgb.perf,
     avg="threshold",
     colorize=TRUE,
     lwd=1,
     main="ROC Curve w/ Thresholds",
     print.cutoffs.at=seq(0, 1, by=0.05),
     text.adj=c(-0.5, 0.5),
     text.cex=0.5)
grid(col="lightgray")
axis(1, at=seq(0, 1, by=0.5))
axis(2, at=seq(0, 1, by=0.5))
abline(v=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
abline(h=c(0.1, 0.3, 0.5, 0.7, 0.9), col="lightgray", lty="dotted")
lines(x=c(0, 1), y=c(0, 1), col="black", lty="dotted")

# End of script

