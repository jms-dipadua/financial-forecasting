library(ggplot2) # for visualization
setwd("/Users/jadalm/Sites/thesis/data/") # my working dir

data_raw_train <- read.csv("working/v5/ge-v5-1.csv", header=TRUE, sep = ',') # get raw train data...where s&p500 prices are
sp500 <- data_raw_train$spClose[2232:2477] # get the relevant period

#should probably re-write all this to use sapply and then i have one "loop" that goes through each of them rather than doing this all "by hand"


xom_raw <- read.csv("outputs/v5/xom-v5-1.csv", header=TRUE, sep = ',')
xom_1 <- xom_raw
xom_1 <- xom_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
xom_1$sp500 <- sp500
xom_1$Date <- as.Date(xom_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
xom_v5_1_plots <- ggplot(xom_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="XOM - Pure Technical Model")
ggsave(file="plots/xom_v5_1_plots.png", plot = last_plot())

xom_raw <- read.csv("outputs/v5/xom-v5-2.csv", header=TRUE, sep = ',')
xom_2 <- xom_raw
xom_2 <- xom_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
xom_2$sp500 <- sp500
xom_2$Date <- as.Date(xom_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
xom_v5_2_plots <- ggplot(xom_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="XOM - Fundamentals Model")
ggsave(file="plots/xom_v5_2_plots.png", plot = last_plot())

xom_raw <- read.csv("outputs/v5/xom-v5-3.csv", header=TRUE, sep = ',')
xom_3 <- xom_raw
xom_3 <- xom_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
xom_3$sp500 <- sp500
xom_3$Date <- as.Date(xom_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
xom_v5_3_plots <- ggplot(xom_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="XOM - Blended Model")
ggsave(file="plots/xom_v5_3_plots.png", plot = last_plot())

cvx_raw <- read.csv("outputs/v5/cvx-v5-1.csv", header=TRUE, sep = ',')
cvx_1 <- cvx_raw
cvx_1 <- cvx_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
cvx_1$sp500 <- sp500
cvx_1$Date <- as.Date(cvx_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
cvx_v5_1_plots <- ggplot(cvx_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="CVX - Pure Technical Model")
ggsave(file="plots/cvx_v5_1_plots.png", plot = last_plot())

cvx_raw <- read.csv("outputs/v5/cvx-v5-2.csv", header=TRUE, sep = ',')
cvx_2 <- cvx_raw
cvx_2 <- cvx_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
cvx_2$sp500 <- sp500
cvx_2$Date <- as.Date(cvx_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
cvx_v5_2_plots <- ggplot(cvx_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="CVX - Fundamentals Model")
ggsave(file="plots/cvx_v5_2_plots.png", plot = last_plot())

cvx_raw <- read.csv("outputs/v5/cvx-v5-3.csv", header=TRUE, sep = ',')
cvx_3 <- cvx_raw
cvx_3 <- cvx_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
cvx_3$sp500 <- sp500
cvx_3$Date <- as.Date(cvx_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
cvx_v5_3_plots <- ggplot(cvx_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="CVX - Blended Model")
ggsave(file="plots/cvx_v5_3_plots.png", plot = last_plot())


# sampling of other securiteis 
msft_raw <- read.csv("outputs/v5/msft-v5-1.csv", header=TRUE, sep = ',')
msft_1 <- msft_raw
msft_1 <- msft_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
msft_1$sp500 <- sp500
msft_1$Date <- as.Date(msft_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
msft_v5_1_plots <- ggplot(msft_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MSFT - Pure Technical Model")
ggsave(file="plots/msft_v5_1_plots.png", plot = last_plot())

msft_raw <- read.csv("outputs/v5/msft-v5-3.csv", header=TRUE, sep = ',')
msft_3 <- msft_raw
msft_3 <- msft_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
msft_3$Date <- as.Date(msft_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
msft_v5_3_plots <- ggplot(msft_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MSFT - Blended Model")
ggsave(file="plots/msft_v5_3_plots.png", plot = last_plot())

# mickie Ds
mcd_raw <- read.csv("outputs/v5/mcd-v5-2.csv", header=TRUE, sep = ',')
mcd_2 <- mcd_raw
mcd_2 <- mcd_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
mcd_2$Date <- as.Date(mcd_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
mcd_v5_2_plots <- ggplot(mcd_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MCD - Fundamnetals Model")
ggsave(file="plots/mcd_v5_2_plots.png", plot = last_plot())

mcd_raw <- read.csv("outputs/v5/mcd-v5-3.csv", header=TRUE, sep = ',')
mcd_3 <- mcd_raw
mcd_3 <- mcd_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
mcd_3$Date <- as.Date(mcd_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
mcd_v5_3_plots <- ggplot(mcd_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MCD - Blended Model")
ggsave(file="plots/mcd_v5_3_plots.png", plot = last_plot())

# start looking at P&L
xom_raw1 <- read.csv("outputs/v5/xom-v5-1_pl.csv", header=TRUE, sep = ',')
xom_raw2 <- read.csv("outputs/v5/xom-v5-2_pl.csv", header=TRUE, sep = ',')
xom_raw3 <- read.csv("outputs/v5/xom-v5-3_pl.csv", header=TRUE, sep = ',')

tgt_raw1 <- read.csv("outputs/v5/tgt-v5-1_pl.csv", header=TRUE, sep = ',')
tgt_raw2 <- read.csv("outputs/v5/tgt-v5-2_pl.csv", header=TRUE, sep = ',')
tgt_raw3 <- read.csv("outputs/v5/tgt-v5-3_pl.csv", header=TRUE, sep = ',')

cvx_raw1 <- read.csv("outputs/v5/cvx-v5-1_pl.csv", header=TRUE, sep = ',')
cvx_raw2 <- read.csv("outputs/v5/cvx-v5-2_pl.csv", header=TRUE, sep = ',')
cvx_raw3 <- read.csv("outputs/v5/cvx-v5-3_pl.csv", header=TRUE, sep = ',')