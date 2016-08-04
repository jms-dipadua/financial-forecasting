library(ggplot2) # for visualization
library(reshape2)
setwd("/Users/jadalm/Sites/thesis/data/") # my working dir

data_raw_train <- read.csv("working/v5/ge-v5-1.csv", header=TRUE, sep = ',') # get raw train data...where s&p500 prices are
sp500 <- data_raw_train$spClose[2232:2477] # get the relevant period

#should probably re-write all this to use sapply and then i have one "loop" that goes through each of them rather than doing this all "by hand"


xom_raw <- read.csv("outputs/v5/xom-v5-1.csv", header=TRUE, sep = ',')
xom_1 <- xom_raw
xom_1 <- xom_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
xom_1$sp500 <- sp500
xom_1$Date <- as.Date(xom_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
xom_1$Year <- format(xom_1$Date, "%Y")
xom_1_2015 <- xom_1[xom_1$Year == 2015, ]
xom_1_2015 <- xom_1_2015[-c(4)] # drop date; messing things up
xom_1_2015_m <- melt(xom_1_2015) # get it into the right shape

# graphs
# all of them
xom_v5_1_plots <- ggplot(xom_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="XOM - Pure Technical Model")
ggsave(file="plots/xom_v5_1_plots.png", plot = last_plot())
xom_1_2015_m_plot <- qplot(factor(variable),value, data=xom_1_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="Exxon Price Distributions, Actual and Forecasted (Technicals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/xom_1_price_distros_2015.png", plot = last_plot())



xom_raw <- read.csv("outputs/v5/xom-v5-2.csv", header=TRUE, sep = ',')
xom_2 <- xom_raw
xom_2 <- xom_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
xom_2$sp500 <- sp500
xom_2$Date <- as.Date(xom_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
xom_2$Year <- format(xom_2$Date, "%Y")
xom_2_2015 <- xom_2[xom_2$Year == 2015, ]
xom_2_2015 <- xom_2_2015[-c(4)] # drop date; messing things up
xom_2_2015_m <- melt(xom_2_2015) # get it into the right shape

# graphs
# all of them
xom_v5_2_plots <- ggplot(xom_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="XOM - Fundamentals Model")
ggsave(file="plots/xom_v5_2_plots.png", plot = last_plot())
xom_2_2015_m_plot <- qplot(factor(variable),value, data=xom_2_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="Exxon Price Distributions, Actual and Forecasted (Fundamentals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/xom_2_price_distros_2015.png", plot = last_plot())

xom_raw <- read.csv("outputs/v5/xom-v5-3.csv", header=TRUE, sep = ',')
xom_3 <- xom_raw
xom_3 <- xom_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
xom_3$sp500 <- sp500
xom_3$Date <- as.Date(xom_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
xom_v5_3_plots <- ggplot(xom_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="XOM - Blended Model")
ggsave(file="plots/xom_v5_3_plots.png", plot = last_plot())

xom_3$Year <- format(xom_3$Date, "%Y")
xom_3_2015 <- xom_3[xom_3$Year == 2015, ]
xom_3_2015 <- xom_3_2015[-c(4)] # drop date; messing things up
xom_3_2015_m <- melt(xom_3_2015) # get it into the right shape
xom_3_2015_m_plot <- qplot(factor(variable),value, data=xom_3_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="Exxon Price Distributions, Actual and Forecasted (Blended)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/xom_3_price_distros_2015.png", plot = last_plot())



cvx_raw <- read.csv("outputs/v5/cvx-v5-1.csv", header=TRUE, sep = ',')
cvx_1 <- cvx_raw
cvx_1 <- cvx_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
cvx_1$sp500 <- sp500
cvx_1$Date <- as.Date(cvx_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
cvx_v5_1_plots <- ggplot(cvx_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="CVX - Pure Technical Model")
ggsave(file="plots/cvx_v5_1_plots.png", plot = last_plot())

cvx_1$Year <- format(cvx_1$Date, "%Y")
cvx_1_2015 <- cvx_1[cvx_1$Year == 2015, ]
cvx_1_2015 <- cvx_1_2015[-c(4)] # drop date; messing things up
cvx_1_2015_m <- melt(cvx_1_2015) # get it into the right shape
cvx_1_2015_m_plot <- qplot(factor(variable),value, data=cvx_1_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="Chevron Price Distributions, Actual and Forecasted (Technicals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/cvx_1_price_distros_2015.png", plot = last_plot())

cvx_raw <- read.csv("outputs/v5/cvx-v5-2.csv", header=TRUE, sep = ',')
cvx_2 <- cvx_raw
cvx_2 <- cvx_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
cvx_2$sp500 <- sp500
cvx_2$Date <- as.Date(cvx_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
cvx_v5_2_plots <- ggplot(cvx_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="CVX - Fundamentals Model")
ggsave(file="plots/cvx_v5_2_plots.png", plot = last_plot())

cvx_2$Year <- format(cvx_2$Date, "%Y")
cvx_2_2015 <- cvx_2[cvx_2$Year == 2015, ]
cvx_2_2015 <- cvx_2_2015[-c(4)] # drop date; messing things up
cvx_2_2015_m <- melt(cvx_2_2015) # get it into the right shape
cvx_2_2015_m_plot <- qplot(factor(variable),value, data=cvx_2_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="Chevron Price Distributions, Actual and Forecasted (Fundamentals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/cvx_2_price_distros_2015.png", plot = last_plot())

cvx_raw <- read.csv("outputs/v5/cvx-v5-3.csv", header=TRUE, sep = ',')
cvx_3 <- cvx_raw
cvx_3 <- cvx_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
cvx_3$sp500 <- sp500
cvx_3$Date <- as.Date(cvx_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
cvx_v5_3_plots <- ggplot(cvx_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="CVX - Blended Model")
ggsave(file="plots/cvx_v5_3_plots.png", plot = last_plot())

cvx_3$Year <- format(cvx_3$Date, "%Y")
cvx_3_2015 <- cvx_3[cvx_3$Year == 2015, ]
cvx_3_2015 <- cvx_3_2015[-c(4)] # drop date; messing things up
cvx_3_2015_m <- melt(cvx_3_2015) # get it into the right shape
cvx_3_2015_m_plot <- qplot(factor(variable),value, data=cvx_3_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="Chevron Price Distributions, Actual and Forecasted (Blended)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/cvx_3_price_distros_2015.png", plot = last_plot())



# sampling of other securiteis 
msft_raw <- read.csv("outputs/v5/msft-v5-1.csv", header=TRUE, sep = ',')
msft_1 <- msft_raw
msft_1 <- msft_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
msft_1$sp500 <- sp500
msft_1$Date <- as.Date(msft_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
msft_v5_1_plots <- ggplot(msft_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MSFT - Pure Technical Model")
ggsave(file="plots/msft_v5_1_plots.png", plot = last_plot())

msft_1$Year <- format(msft_1$Date, "%Y")
msft_1_2015 <- msft_1[msft_1$Year == 2015, ]
msft_1_2015 <- msft_1_2015[-c(4)] # drop date; messing things up
msft_1_2015_m <- melt(msft_1_2015) # get it into the right shape
msft_1_2015_m_plot <- qplot(factor(variable),value, data=msft_1_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="MSFT Price Distributions, Actual and Forecasted (Technicals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/msft_1_price_distros_2015.png", plot = last_plot())

msft_raw <- read.csv("outputs/v5/msft-v5-3.csv", header=TRUE, sep = ',')
msft_3 <- msft_raw
msft_3 <- msft_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
msft_3$Date <- as.Date(msft_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
msft_v5_3_plots <- ggplot(msft_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MSFT - Blended Model")
ggsave(file="plots/msft_v5_3_plots.png", plot = last_plot())

msft_3$Year <- format(msft_3$Date, "%Y")
msft_3_2015 <- msft_3[msft_3$Year == 2015, ]
msft_3_2015 <- msft_3_2015[-c(4)] # drop date; messing things up
msft_3_2015_m <- melt(msft_3_2015) # get it into the right shape
msft_3_2015_m_plot <- qplot(factor(variable),value, data=msft_3_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="MSFT Price Distributions, Actual and Forecasted (Blended)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/msft_3_price_distros_2015.png", plot = last_plot())


# mickie Ds
mcd_raw <- read.csv("outputs/v5/mcd-v5-2.csv", header=TRUE, sep = ',')
mcd_2 <- mcd_raw
mcd_2 <- mcd_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
mcd_2$Date <- as.Date(mcd_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
mcd_v5_2_plots <- ggplot(mcd_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MCD - Fundamnetals Model")
ggsave(file="plots/mcd_v5_2_plots.png", plot = last_plot())

mcd_2$Year <- format(mcd_2$Date, "%Y")
mcd_2_2015 <- mcd_2[mcd_2$Year == 2015, ]
mcd_2_2015 <- mcd_2_2015[-c(4)] # drop date; messing things up
mcd_2_2015_m <- melt(mcd_2_2015) # get it into the right shape
mcd_2_2015_m_plot <- qplot(factor(variable),value, data=mcd_2_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="MCD Price Distributions, Actual and Forecasted (Fundamentals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/mcd_2_price_distros_2015.png", plot = last_plot())


mcd_raw <- read.csv("outputs/v5/mcd-v5-3.csv", header=TRUE, sep = ',')
mcd_3 <- mcd_raw
mcd_3 <- mcd_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
mcd_3$Date <- as.Date(mcd_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
mcd_v5_3_plots <- ggplot(mcd_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="MCD - Blended Model")
ggsave(file="plots/mcd_v5_3_plots.png", plot = last_plot())

mcd_3$Year <- format(mcd_3$Date, "%Y")
mcd_3_2015 <- mcd_3[mcd_3$Year == 2015, ]
mcd_3_2015 <- mcd_3_2015[-c(4)] # drop date; messing things up
mcd_3_2015_m <- melt(mcd_3_2015) # get it into the right shape
mcd_3_2015_m_plot <- qplot(factor(variable),value, data=mcd_3_2015_m, geom="boxplot", fill=factor(variable)) + labs(title="MCD Price Distributions, Actual and Forecasted (Blended)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/mcd_3_price_distros_2015.png", plot = last_plot())



ge_raw_train <- read.csv("working/v5/ge-v5-1.csv", header=TRUE, sep = ',') 


ge_2_train <- ge_raw_train
colnames(ge_2_train) <- c('id', 'Date', 'Open','High', 'Low', 'Close', 'Volume', 'SMA.5', 'SMA.15', 'SMA.50', 'SMA.200', "WMA.10", "WMA.30", "WMA.100", "WMA.200", "CCI.20", "RSI.14", "TresYield", "BrentCrude", "CivilianEmp.", "UnEmp.Rate", "CPI.Urban", "FedDebtToGDP", "HousingStarts", "InitialJoblessClaims", "LIBOR", "Per.Cons.Expend.", "PrivSaveRate", "RealGDP",  "SP500", "USD.Euro", "TotalRevenue", "NetIncome", "EPS", "TotalAssets", "TotalLiabilities",  "FreeCashFlow", "ProfitMargin", "P.E.Ratio")
ge_2_train$Date <- as.Date(ge_2_train$Date, "%m/%d/%Y")
ge_2_train$Year <- format(ge_2_train$Date, "%Y")

price_dist <- qplot(factor(Year), Close, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="GE, Closing Price by Year", fill="Year", x="", y="Closing Price, USD")
ggsave(file="plots/ge-close-by-year.png", plot = last_plot())

price_dist <- qplot(factor(Year), BrentCrude, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Brent Crude, USD per BBL by Year", fill="Year", x="", y="Closing Price, USD")
ggsave(file="plots/oil-close-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), TresYield, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="10-Year Tres. Yield", fill="Year", x="", y="Percentage Yield")
ggsave(file="plots/tres-yield-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), CivilianEmp., data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Civilian Labor Participation Rate", fill="Year", x="", y="Percentage Working Age Employed")
ggsave(file="plots/civilian-employed-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), UnEmp.Rate, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Unemployment Rate", fill="Year", x="", y="Percentage Working Age Seeking Employment")
ggsave(file="plots/unemployed-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), CPI.Urban, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Consumper Price Index, Urban Consumers", fill="Year", x="", y="Average Change in Consumer Price Index")
ggsave(file="plots/cpi-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), FedDebtToGDP, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Ratio of Federal Debt to (Real) GDP", fill="Year", x="", y="Percentage, Federal Debt to GDP")
ggsave(file="plots/fed-debt-to-gdp-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), HousingStarts, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="New (Private Dwelling) Housing Starts by Year", fill="Year", x="", y="Thousands of Unitis, Private Dwellings")
ggsave(file="plots/housing-starts-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), InitialJoblessClaims, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="New Jobless Claims, 4-week Moving Average", fill="Year", x="", y="Number of New Claims")
ggsave(file="plots/jobless-claims-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), LIBOR, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="1-Month London Interbank Offered Rate, in USD", fill="Year", x="", y="LIBOR")
ggsave(file="plots/libor-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), Per.Cons.Expend., data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Personal Consumption Expenditures", fill="Year", x="", y="Billions of USD")
ggsave(file="plots/PCE-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), PrivSaveRate, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Personal Savings Rate", fill="Year", x="", y="Percentage of Disposable Income")
ggsave(file="plots/PSR-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), RealGDP, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Change in (Real) Gross Domestic Product", fill="Year", x="", y="Percentage Change")
ggsave(file="plots/rGDP-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), SP500, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="S&P500 Index", fill="Year", x="", y="Weighted Average, Market Capitalization at Close, USD")
ggsave(file="plots/SP500-by-year.png", plot = last_plot())

distro <- qplot(factor(Year), USD.Euro, data=ge_2_train, geom="boxplot", fill=ge_2_train$Year) + labs(title="Exchange Rate, USD-to-Euro", fill="Year", x="", y="Euros Per USD")
ggsave(file="plots/usd-euro-by-year.png", plot = last_plot())


histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = BrentCrude)) + labs(title = "Brent Crude, USD per BBL by Year")
ggsave(file="plots/brentHisto_plots.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y =TresYield)) + labs(title = "10-Year Tres. Yield", y="Percentage Yield")
ggsave(file="plots/10yrTresHisto_plots.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y =CivilianEmp.)) + labs(title = "Civilian Labor Participation Rate", x="", y="Percentage Working Age Employed")
ggsave(file="plots/CivilianEmpHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y =UnEmp.Rate)) + labs(title="Unemployment Rate", x="", y="Percentage Working Age Seeking Employment")
ggsave(file="plots/UnEmp.RateHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y =CPI.Urban)) + labs(title="Consumper Price Index, Urban Consumers", x="", y="Average Change in Consumer Price Index")
ggsave(file="plots/CPI.UrbanHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = FedDebtToGDP)) + labs(title="Ratio of Federal Debt to (Real) GDP", x="", y="Percentage, Federal Debt to GDP")
ggsave(file="plots/FedDebtToGDPHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = FedDebtToGDP)) + labs(title="New (Private Dwelling) Housing Starts by Year", x="", y="Thousands of Unitis, Private Dwellings")
ggsave(file="plots/HousingStartsHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = InitialJoblessClaims)) + labs(title="New Jobless Claims, 4-week Moving Average", x="", y="Number of New Claims")
ggsave(file="plots/InitialJoblessClaimsHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = LIBOR)) + labs(title="1-Month London Interbank Offered Rate, in USD", x="", y="LIBOR")
ggsave(file="plots/LIBORHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = Per.Cons.Expend.)) + labs(title="Personal Consumption Expenditures", x="", y="Billions of USD")
ggsave(file="plots/Per.Cons.Expend.Histo.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = PrivSaveRate)) + labs(title="Personal Savings Rate", fill="Year", y="Percentage of Disposable Income")
ggsave(file="plots/PrivSaveRate.Histo.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = RealGDP)) + labs(title="Change in (Real) Gross Domestic Product", x="", y="Percentage Change")
ggsave(file="plots/RealGDPHisto.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = SP500)) + labs(title="S&P500 Index", x="", y="Weighted Average, Market Capitalization at Close, USD")
ggsave(file="plots/SP500Histo.png", plot = last_plot())

histo <- ggplot(ge_2_train, aes(x=Date)) + geom_line(aes(y = USD.Euro)) + labs(title="Exchange Rate, USD-to-Euro", x="", y="Euros Per USD")
ggsave(file="plots/USD.EuroHisto.png", plot = last_plot())


# time permitting: 
#cbbPalette <- c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")

# To use for fills, add
#scale_fill_manual(values=cbPalette)

svr_rmse <- read.csv("RMSE-by-Experiment-SVR-v5.csv", header=TRUE, sep = ',')
colnames(svr_rmse) <- c('Company', 'Technicals', 'Fundamentals', 'Blended')
svr_rmse_m <- melt(svr_rmse,id.vars = 1) # get it into the right shape

ann_rmse <- read.csv("RMSE-by-Experiment-ANN-v5.csv", header=TRUE, sep = ',')
colnames(ann_rmse) <- c('Company', 'Technicals', 'Fundamentals', 'Blended')
ann_rmse_m <- melt(ann_rmse,id.vars = 1) # get it into the right shape

rmse_plot <- ggplot(svr_rmse_m,aes(x=variable,y=value,fill=Company, color = Company)) + geom_bar(stat="identity",position="dodge") + scale_fill_discrete(name="Company") + xlab("Experiment Type") + ylab("RMSE") + labs(title="SVR RMSE per Model, by Company") + coord_flip()
ggsave(file="plots/svr_rmse_all.png", plot = last_plot())

rmse_plot <- ggplot(ann_rmse_m,aes(x=variable,y=value,fill=Company, color = Company)) + geom_bar(stat="identity",position="dodge") + scale_fill_discrete(name="Company") + xlab("Experiment Type") + ylab("RMSE") + labs(title="ANN RMSE per Model, by Company") + coord_flip()
ggsave(file="plots/ann_rmse_all.png", plot = last_plot())


att_raw <- read.csv("outputs/v5/t-v5-1.csv", header=TRUE, sep = ',')
att_1 <- att_raw
att_1 <- att_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
att_1$Date <- as.Date(att_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
att_1_v5_1_plots <- ggplot(att_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="AT&T - Pure Technical Model")
ggsave(file="plots/att1_v5_1_plots.png", plot = last_plot())

att_1$Year <- format(att_1$Date, "%Y")
att_1_2015 <- att_1[att_1$Year == 2015, ]
att_1_2015_1 <- att_1_2015[-c(4)] # drop date; messing things up
att_1_2015_m2 <- melt(att_1_2015_1) # get it into the right shape
att_1_2015_m_plot <- qplot(factor(variable),value, data=att_1_2015_m2, geom="boxplot", fill=factor(variable)) + labs(title="AT&T Price Distributions, Actual and Forecasted (Technicals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/t_1_price_distros_2015.png", plot = last_plot())

att_raw <- read.csv("outputs/v5/t-v5-2.csv", header=TRUE, sep = ',')
att_2 <- att_raw
att_2 <- att_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
att_2$Date <- as.Date(att_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
att_v5_2_plots <- ggplot(att_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="AT&T - Fundamentals Model")
ggsave(file="plots/att_v5_2_plots.png", plot = last_plot())

att_2$Year <- format(att_2$Date, "%Y")
att_2_2015 <- att_2[att_2$Year == 2015, ]
att_2_2015_2 <- att_2_2015[-c(4)] # drop date; messing things up
att_2_2015_m2 <- melt(att_2_2015_2) # get it into the right shape
att_2_2015_m_plot <- qplot(factor(variable),value, data=att_2_2015_m2, geom="boxplot", fill=factor(variable)) + labs(title="AT&T Price Distributions, Actual and Forecasted (Technicals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/t_2_price_distros_2015.png", plot = last_plot())

att_raw <- read.csv("outputs/v5/t-v5-3.csv", header=TRUE, sep = ',')
att_3 <- att_raw
att_3 <- att_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
att_3$Date <- as.Date(att_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
att_v5_3_plots <- ggplot(att_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="AT&T - Blended Model")
ggsave(file="plots/att_v5_3_plots.png", plot = last_plot())

att_3$Year <- format(att_3$Date, "%Y")
att_3_2015 <- att_3[att_3$Year == 2015, ]
att_3_2015_2 <- att_3_2015[-c(4)] # drop date; messing things up
att_3_2015_m2 <- melt(att_3_2015_2) # get it into the right shape
att_3_2015_m_plot <- qplot(factor(variable),value, data=att_3_2015_m2, geom="boxplot", fill=factor(variable)) + labs(title="AT&T Price Distributions, Actual and Forecasted (Blended)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/t_3_price_distros_2015.png", plot = last_plot())


f_raw <- read.csv("outputs/v5/f-v5-1.csv", header=TRUE, sep = ',')
f_1 <- f_raw
f_1 <- f_1[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
f_1$Date <- as.Date(f_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

f_1_v5_1_plots <- ggplot(f_1, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="Ford - Pure Technical Model")
ggsave(file="plots/f1_v5_1_plots.png", plot = last_plot())

f_1$Year <- format(f_1$Date, "%Y")
f_1_2015 <- f_1[f_1$Year == 2015, ]
f_1_2015_1 <- f_1_2015[-c(4)] # drop date; messing things up
f_1_2015_m2 <- melt(f_1_2015_1) # get it into the right shape
f_1_2015_m_plot <- qplot(factor(variable),value, data=f_1_2015_m2, geom="boxplot", fill=factor(variable)) + labs(title="Ford Price Distributions, Actual and Forecasted (Technicals)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/f_1_price_distros_2015.png", plot = last_plot())

f_raw <- read.csv("outputs/v5/f-v5-2.csv", header=TRUE, sep = ',')
f_2 <- f_raw
f_2 <- f_2[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
f_2$Date <- as.Date(f_2$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format

# graphs
# all of them
f_v5_2_plots <- ggplot(f_2, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="Ford - Fundamentals Model")
ggsave(file="plots/f_v5_2_plots.png", plot = last_plot())

f_2$Year <- format(f_3$Date, "%Y")
f_3_2015 <- f_3[f_3$Year == 2015, ]
f_3_2015_2 <- f_3_2015[-c(4)] # drop date; messing things up
f_3_2015_m2 <- melt(f_3_2015_2) # get it into the right shape
f_3_2015_m_plot <- qplot(factor(variable),value, data=f_3_2015_m2, geom="boxplot", fill=factor(variable)) + labs(title="Ford Price Distributions, Actual and Forecasted (Blended)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/f_3_price_distros_2015.png", plot = last_plot())

f_raw <- read.csv("outputs/v5/f-v5-3.csv", header=TRUE, sep = ',')
f_3 <- f_raw
f_3 <- f_3[-c(1,5,6)] # gets rid of ids and purchase decisions (keeps dates)
f_3$Date <- as.Date(f_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
f_3$Year <- format(f_3$Date, "%Y")
f_3_2015 <- f_3[f_3$Year == 2015, ]

# graphs
# all of them
f_v5_3_plots <- ggplot(f_3, aes(x=Date)) + geom_line(aes(y = Actual, color = "Actual Close")) + geom_line(aes(y = SVM, color = "SVR")) + geom_line(aes(y = ANN, color = "Neural Network")) + ylab("Price, USD") + labs(title="Ford - Blended Model")
ggsave(file="plots/f_v5_3_plots.png", plot = last_plot())

# how about boxplots of prices
# actual, svm, ann_rmse
f_3_2015_2 <- f_3_2015[-c(4)] # drop date; messing things up
f_3_2015_m2 <- melt(f_3_2015_2) # get it into the right shape
f_3_2015_m_plot <- qplot(factor(variable),value, data=f_3_2015_m2, geom="boxplot", fill=factor(variable)) + labs(title="Ford Price Distributions, Actual and Forecasted (Blended)", y="USD", x="Model") + guides(fill=guide_legend(title="Forecast by Type")) + coord_flip()
ggsave(file="plots/f_3_price_distros_2015.png", plot = last_plot())



# start looking at P&L
# ... NOT SURE THIS WILL BE INTERESTING...
xom_raw1 <- read.csv("outputs/v5/xom-v5-1_pl.csv", header=TRUE, sep = ',')
xom_raw2 <- read.csv("outputs/v5/xom-v5-2_pl.csv", header=TRUE, sep = ',')
xom_raw3 <- read.csv("outputs/v5/xom-v5-3_pl.csv", header=TRUE, sep = ',')

tgt_raw1 <- read.csv("outputs/v5/tgt-v5-1_pl.csv", header=TRUE, sep = ',')
tgt_raw2 <- read.csv("outputs/v5/tgt-v5-2_pl.csv", header=TRUE, sep = ',')
tgt_raw3 <- read.csv("outputs/v5/tgt-v5-3_pl.csv", header=TRUE, sep = ',')

cvx_raw1 <- read.csv("outputs/v5/cvx-v5-1_pl.csv", header=TRUE, sep = ',')
cvx_raw2 <- read.csv("outputs/v5/cvx-v5-2_pl.csv", header=TRUE, sep = ',')
cvx_raw3 <- read.csv("outputs/v5/cvx-v5-3_pl.csv", header=TRUE, sep = ',')

# more technicals - simple graphs of SMA/WMA, close, rsi & CCI
xom_raw <- read.csv("working/v5/xom-v5-1.csv", header=TRUE, sep = ',')
xom_2 <- xom_raw
xom_3 <- xom_2[c('Date', 'Close', 'SMA.5', 'SMA.15', 'SMA.50', 'SMA.200', 'WMA.10', 'WMA.30', 'WMA.100', 'WMA.200', 'cci.20', 'rsi.14')]
xom_3$Date <- as.Date(xom_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
xom_3$Year <- format(xom_3$Date, "%Y")
xom_3_2015 <- xom_3[xom_3$Year == 2015, ]

xom_v5_t1_plots <- ggplot(xom_3_2015, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = SMA.50, color = "SMA-50")) + geom_line(aes(y = WMA.30, color = "WMA-30")) + geom_line(aes(y = WMA.100, color = "WMA-100")) + ylab("Price, USD") + xlab("") + labs(title="Exxon, 2015 - Close Price and Sample Moving Averages")
ggsave(file="plots/xom_v5_t1_plots.png", plot = last_plot())

mcd_raw <- read.csv("working/v5/mcd-v5-1.csv", header=TRUE, sep = ',')
mcd_2 <- mcd_raw
mcd_3 <- mcd_2[c('Date', 'Close', 'SMA.5', 'SMA.15', 'SMA.50', 'SMA.200', 'WMA.10', 'WMA.30', 'WMA.100', 'WMA.200', 'cci.20', 'rsi.14')]
mcd_3$Date <- as.Date(mcd_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
mcd_3$Year <- format(mcd_3$Date, "%Y")
mcd_3_2015 <- mcd_3[mcd_3$Year == 2015, ]

mcd_v5_t1_plots <- ggplot(mcd_3_2015, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = SMA.50, color = "SMA-50")) + geom_line(aes(y = WMA.30, color = "WMA-30")) + geom_line(aes(y = WMA.100, color = "WMA-100")) + ylab("Price, USD") + xlab("") + labs(title="McDonalds, 2015 - Close Price and Sample Moving Averages")
ggsave(file="plots/mcd_v5_t1_plots.png", plot = last_plot())

t_raw <- read.csv("working/v5/t-v5-1.csv", header=TRUE, sep = ',')
t_2 <- t_raw
t_3 <- t_2[c('Date', 'Close', 'SMA.5', 'SMA.15', 'SMA.50', 'SMA.200', 'WMA.10', 'WMA.30', 'WMA.100', 'WMA.200', 'cci.20', 'rsi.14')]
t_3$Date <- as.Date(t_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
t_3$Year <- format(t_3$Date, "%Y")
t_3_2015 <- t_3[t_3$Year == 2015, ]

t_v5_t1_plots <- ggplot(t_3_2015, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = SMA.50, color = "SMA-50")) + geom_line(aes(y = WMA.30, color = "WMA-30")) + geom_line(aes(y = WMA.200, color = "WMA-200")) + ylab("Price, USD") + xlab("") + labs(title="AT&T, 2015 - Close Price and Sample Moving Averages")
ggsave(file="plots/t_v5_t1_plots.png", plot = last_plot())


msft_raw <- read.csv("working/v5/msft-v5-1.csv", header=TRUE, sep = ',')
msft_2 <- msft_raw
msft_3 <- msft_2[c('Date', 'Close', 'SMA.5', 'SMA.15', 'SMA.50', 'SMA.200', 'WMA.10', 'WMA.30', 'WMA.100', 'WMA.200', 'cci.20', 'rsi.14')]
msft_3$Date <- as.Date(msft_3$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
msft_3$Year <- format(msft_3$Date, "%Y")
msft_3_2015 <- msft_3[msft_3$Year == 2015, ]

msft_v5_t1_plots <- ggplot(msft_3_2015, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = SMA.50, color = "SMA-50")) + geom_line(aes(y = WMA.30, color = "WMA-30")) + geom_line(aes(y = WMA.200, color = "WMA-200")) + ylab("Price, USD") + xlab("") + labs(title="Microsoft, 2015 - Close Price and Sample Moving Averages")
ggsave(file="plots/msft_v5_t1_plots.png", plot = last_plot())


msft_4 <- msft_2[c('Date', 'Close', "total_revenue", "net_income", "EPS", "total_assets", "total_liabilities", "free_cash_flow", "profit_margin", "PE")]
msft_4$Date <- as.Date(msft_4$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
msft_4$Year <- format(msft_4$Date, "%Y")
msft_4_2015 <- msft_3[msft_4$Year == 2015, ]

msft_v5_t1_plots <- ggplot(msft_4, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = total_revenue, color = "Total Revenue")) + geom_line(aes(y = net_income, color = "Net Income")) + geom_line(aes(y = total_assets, color = "Total Assets")) + geom_line(aes(y = total_liabilities, color = "Total Liabilities")) + ylab("Price, Billions USD (except Close)") + xlab("") + labs(title="Microsoft - Close Price with Example Fundamnetals")
ggsave(file="plots/msft_v5_micros_plots.png", plot = last_plot())


ba_raw <- read.csv("working/v5/ba-v5-1.csv", header=TRUE, sep = ',')
ba_2 <- ba_raw
ba_4 <- ba_2[c('Date', 'Close', "total_revenue", "net_income", "EPS", "total_assets", "total_liabilities", "free_cash_flow", "profit_margin", "PE")]
ba_4$Date <- as.Date(ba_4$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
ba_4$Year <- format(ba_4$Date, "%Y")
ba_4_2015 <- ba_4[ba_4$Year == 2015, ]

ba_v5_t1_plots <- ggplot(ba_4, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = total_revenue, color = "Total Revenue")) + geom_line(aes(y = net_income, color = "Net Income")) + geom_line(aes(y = total_assets, color = "Total Assets")) + geom_line(aes(y = total_liabilities, color = "Total Liabilities")) + ylab("Price, Billions USD (except Close)") + xlab("") + labs(title="Boeing - Close Price with Example Fundamnetals")
ggsave(file="plots/ba_v5_micros_plots.png", plot = last_plot())

orcl_raw <- read.csv("working/v5/orcl-v5-1.csv", header=TRUE, sep = ',')
orcl_2 <- orcl_raw
orcl_4 <- orcl_2[c('Date', 'Close', "total_revenue", "net_income", "EPS", "total_assets", "total_liabilities", "free_cash_flow", "profit_margin", "PE")]
orcl_4$Date <- as.Date(orcl_4$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
orcl_4$Year <- format(orcl_4$Date, "%Y")
orcl_4_2015 <- orcl_4[orcl_4$Year == 2015, ]

orcl_v5_t1_plots <- ggplot(orcl_4, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = total_revenue, color = "Total Revenue")) + geom_line(aes(y = net_income, color = "Net Income")) + geom_line(aes(y = total_assets, color = "Total Assets")) + geom_line(aes(y = total_liabilities, color = "Total Liabilities")) + ylab("Price, Billions USD (except Close)") + xlab("") + labs(title="Oracle - Close Price with Example Fundamnetals")
ggsave(file="plots/orcl_v5_micros_plots.png", plot = last_plot())

wmt_raw <- read.csv("working/v5/wmt-v5-1.csv", header=TRUE, sep = ',')
wmt_2 <- wmt_raw
wmt_4 <- wmt_2[c('Date', 'Close', "total_revenue", "net_income", "EPS", "total_assets", "total_liabilities", "free_cash_flow", "profit_margin", "PE")]
wmt_4$Date <- as.Date(wmt_4$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
wmt_4$Year <- format(wmt_4$Date, "%Y")
wmt_4_2015 <- wmt_4[wmt_4$Year == 2015, ]

wmt_v5_t1_plots <- ggplot(wmt_4, aes(x=Date)) + geom_line(aes(y = Close, color = "Actual Close")) + geom_line(aes(y = total_revenue, color = "Total Revenue")) + geom_line(aes(y = net_income, color = "Net Income")) + geom_line(aes(y = total_assets, color = "Total Assets")) + geom_line(aes(y = total_liabilities, color = "Total Liabilities")) + ylab("Price, Billions USD (except Close)") + xlab("") + labs(title="Walmart - Close Price with Example Fundamnetals")
ggsave(file="plots/wmt_v5_micros_plots.png", plot = last_plot())


# focus on oil, s&p and xom (as example)
xom_raw <- read.csv("working/v5/xom-v5-1.csv", header=TRUE, sep = ',')
xom_1 <- xom_raw
xom_1$Date <- as.Date(xom_1$Date, format = "%m/%d/%Y") # coerse the date into the "r-ready" format
xom_1$Year <- format(xom_1$Date, "%Y")

xom_sp500_oil <- ggplot(xom_1, aes(x=Date)) + geom_line(aes(y = spClose, color = "S&P 500")) + geom_line(aes(y = DCOILBRENTEU, color = "Brent Crude")) + geom_line(aes(y = Close, color = "Exxon Closing Price")) + ylab("USD Price") + xlab("") + labs(title="Equities Markets: S&P500, XOM and Brent Crude")
ggsave(file="plots/xom_sp500_oil.png", plot = last_plot())

sp500_1 <- xom_1[, c('Date', 'spClose')]
sp500_1 <- ggplot(xom_1, aes(x=Date)) + geom_line(aes(y = spClose)) + ylab("USD") + xlab("") + labs(title="Equities Markets, 2006 - 2015: S&P500")
ggsave(file="plots/sp500_1.png", plot = last_plot())