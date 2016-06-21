library(ggplot2) # for visualization
setwd("/Users/jadalm/Sites/thesis/data/") # my working dir

data_raw_train <- read.csv("working/v5/ge-v5-1.csv", header=TRUE, sep = ',') # get raw train data...where s&p500 prices are
sp500 <- data_raw_train$spClose[2232:2477] # get the relevant period

xom_raw <- read.csv("outputs/v5/xom-v5-1.csv", header=TRUE, sep = ',')
xom_1 <- xom_raw
xom_1 <- xom_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
xom_1$sp500 <- sp500
xom_1$Date <- as.Date(xom_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

ggplot(xom_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD")


  geom_line(aes(y = var1, colour = "var1"))