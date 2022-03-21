#Zhaocheng Li
#CIS number: nwds29


#r Visualization of Data
install.packages('skimr')
library("skimr")
dataset = read.csv('./heart_failure.csv')
skim(dataset)
skimr::skim(dataset)
#test = skim(dataset)
#write.csv(test, './test.csv')
#install.packages('DataExplorer')
DataExplorer::plot_bar(dataset, ncol = 3, by = "fatal_mi")
DataExplorer::plot_histogram(dataset, ncol = 3)
DataExplorer::plot_boxplot(dataset, by = "fatal_mi", 
                           ncol = 3)


#r Preparation of Dataset:
install.packages('caret')
library(caret)
# rename the dataset
dataset = read.csv('./heart_failure.csv')
dataset$fatal_mi = as.factor(dataset$fatal_mi)
# create a list of 80% of the rows in the original dataset we can use for training
train_index = createDataPartition(dataset$fatal_mi, p=0.80, list=FALSE)
# select 20% of the data for validation
validation = dataset[-train_index,]
# use the remaining 80% of data to training and testing the models
trainData = dataset[train_index,]
summary(trainData)
summary(validation)


install.packages('randomForest')
install.packages('kernlab')
install.packages('e1071')

#Normalize the Dataset
preProcValues = preProcess(trainData, method = c("center", "scale"))
trainData = predict(preProcValues, trainData)
summary(trainData)
validData = predict(preProcValues, validation)
summary(validData)



#Cross-Validation and Model Construction
# (10-fold cross validation)
Control = trainControl(method="cv", number=10)
metric = "Accuracy"

# SVM
library("e1071")
set.seed(7)
Grid =  expand.grid(C = c(seq(1,10,2)*0.1, seq(1,10,2)))
fit.svm = train(fatal_mi~., data=trainData, 
                method="svmLinear", metric=metric, 
                trControl=Control, tuneGrid = Grid)

# Random Forest
set.seed(7)
Grid = expand.grid(mtry=1:4)
fit.rf = train(fatal_mi~., data=trainData, method="rf", 
               metric=metric, 
               trControl=Control, tuneGrid = Grid)


#Model Comparison
results = resamples(list(svm=fit.svm, rf=fit.rf))
summary(results)
# compare accuracy of models
dotplot(results)



#r Parameters Tuning for Random Forest Model
library(randomForest)
customRF = list(type = "Classification",
                 library = "randomForest",
                 loop = NULL)

customRF$parameters = data.frame(parameter = c("mtry", "ntree"),
                                  class = rep("numeric", 2),
                                  label = c("mtry", "ntree"))

customRF$grid = function(x, y, len = NULL, search = "grid") {}

customRF$fit = function(x, y, wts, param, lev, last, weights, classProbs) {
  randomForest(x, y,
               mtry = param$mtry,
               ntree=param$ntree)
}

#Predict label
customRF$predict = function(modelFit, newdata, 
                            preProc = NULL, submodels = NULL)
   predict(modelFit, newdata)

#Predict prob
customRF$prob = function(modelFit, newdata, preProc = NULL, 
                         submodels = NULL)
   predict(modelFit, newdata, type = "prob")

customRF$sort = function(x) x[order(x[,1]),]
customRF$levels = function(x) x$classes

set.seed(7) 
Grid = expand.grid(mtry=3:12, ntree=seq(100,600,100))
fit.rf = train(fatal_mi~., data=trainData, method=customRF, 
               metric=metric, trControl=control, 
               tuneGrid = Grid)
fit.rf
plot(fit.rf)

#Comparison of The Prediction Accuracy After Parameter's Tuning
predictions = predict(fit.rf, validData)
confusionMatrix(predictions, validData$fatal_mi)


