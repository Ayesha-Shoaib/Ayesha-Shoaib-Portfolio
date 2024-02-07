# Ayesha's Data Analysis Portfolio

## Time Series Forecasting with Attention Mechanisms using NFLX data

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from attention import Attention  # Assuming the Attention layer is implemented in a separate file

# Load the dataset
data = pd.read_csv('NFLX.csv')

# Convert 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Set 'Date' column as index
data.set_index('Date', inplace=True)

# Sort the index
data.sort_index(inplace=True)

# Display the first few rows of the dataset
print(data.head())

# Check for missing values
print(data.isnull().sum())

# Plot the closing price of Netflix stock
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Close Price')
plt.title('Netflix Stock Closing Price')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()

# Extract 'Close' prices for normalization
close_price = data[['Close']].values

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
close_price_scaled = scaler.fit_transform(close_price)

# Define the number of timesteps for LSTM
timesteps = 30

# Prepare the data for LSTM
X, y = [], []
for i in range(timesteps, len(close_price_scaled)):
    X.append(close_price_scaled[i - timesteps:i, 0])
    y.append(close_price_scaled[i, 0])
X, y = np.array(X), np.array(y)

# Reshape the data for LSTM input
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Build the LSTM model with attention mechanism
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
    Attention(),
    Dense(units=1)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=50, batch_size=32)

# Predict the closing prices
predictions = model.predict(X)

# Inverse transform the predictions
predictions = scaler.inverse_transform(predictions)

# Plot the actual and predicted closing prices
plt.figure(figsize=(12, 6))
plt.plot(data.index[timesteps:], data['Close'][timesteps:], label='Actual Close Price', color='blue')
plt.plot(data.index[timesteps:], predictions, label='Predicted Close Price', color='red')
plt.title('Netflix Stock Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()
```
  Open        High         Low       Close   Adj Close  \
Date                                                                     
2023-02-01  353.859985  365.390015  349.910004  361.989990  361.989990   
2023-02-02  365.160004  368.320007  358.429993  366.890015  366.890015   
2023-02-03  359.079987  379.429993  359.000000  365.899994  365.899994   
2023-02-06  363.640015  368.450012  360.679993  361.480011  361.480011   
2023-02-07  358.510010  364.179993  354.179993  362.950012  362.950012   

             
Date        Volume        
2023-02-01  8005200  
2023-02-02  7857000  
2023-02-03  9402000  
2023-02-06  4994900  
2023-02-07  6289400  
Open         0
High         0
Low          0
Close        0
Adj Close    0
Volume       0
dtype: int64

<img width="281" alt="NFLX close price" src="https://github.com/Ayesha-Shoaib/Ayesha-Shoaib-Portfolio/assets/158636211/120aad9f-5a55-4672-af38-b4d01f6e05c9">

<img width="268" alt="NFLX predicted close price" src="https://github.com/Ayesha-Shoaib/Ayesha-Shoaib-Portfolio/assets/158636211/22f166a2-2be1-4317-8df5-e1d913d7222e">

## Portfolio Analysis

In this Analysis, I focused on collecting and analyzing data for a Canadian investor's portfolio, comprising a riskless asset (T-bill ETF) and a risky asset (S&P500 stock). Historical data was gathered for the Canadian T-bill ETF and S&P500 stock index from May 19, 2000, to June 6, 2018. Additionally, CAD/USD exchange rates were utilized to convert S&P500 returns to CAD.

For the T-bill ETF, various metrics were calculated, including gross annual bond yield, log returns, deannualized bond yield, and daily percentage change. Graphs depicting the daily net bond yield and log returns were plotted, and averages and standard deviations of returns were computed.

For the S&P500, calculations included daily net return in USD and its conversion to CAD, as well as the corresponding gross net return. Graphs for daily net return and log returns were generated, along with analyses for the S&P500 ETF stock index.

I then delved into constructing different weighted portfolios. The initial steps involved computing historical averages, variances, and standard deviations for the risky asset. Subsequently, various weighted portfolios, including equal weights and dynamically re-weighted portfolios, were created and analyzed. Time series graphs were plotted, and the Sharpe Ratio was calculated for portfolio evaluation.

The fourth part aimed at constructing an optimal mean-variance portfolio for investors with varying risk aversions. Excess returns were computed for different risk aversion levels, revealing that high-risk-averse individuals tended to allocate less weight to the risky asset. Mean, variance and Sharpe Ratio were determined for each investor, and portfolio returns and values over time were visualized.

The final stage centered on testing for a unit root in the S&P500 risky asset's price and return. Augmented Dickey-Fuller (ADF) tests were employed to assess the presence of a unit root. The results indicated a unit root in the price series, leading to the adoption of an ARIMA model for compensation.

For daily returns, ADF tests were conducted to evaluate the presence of a random walk. Results suggested an absence of significant evidence for a random walk but indicated the lack of a trend or drift. Consequently, an ARMA model was chosen, emphasizing autoregressive and weighted moving average processes.

The following scripts follow R and Python as demonstration.

### R:
```
install.packages(c("tidyquant", "ggplot2", "TTR", "Metrics", "tseries", "forecast"))

library(tidyquant)
library(ggplot2)
library(TTR)
library(Metrics)
library(tseries)
library(forecast)

# Load data using tidyquant
symbols <- c("CADUSD=X", "^IRX", "^GSPC")
start_date <- "2000-05-19"
end_date <- "2018-06-06"

# Fetch data
financial_data <- tq_get(symbols, from = start_date, to = end_date)

# Data Preprocessing
financial_data <- financial_data %>%
  tq_transmute(select = adjusted,
               mutate_fun = periodReturn,
               period = "daily",
               col_rename = "return") %>%
  tk_tbl()

# Calculate log returns, deannualized bond yield, and other required metrics
financial_data <- financial_data %>%
  mutate(log_returns = log(1 + return),
         deannualized_yield = return / 253)

# Portfolio Analysis
weights <- c(0.5, 0.5)
financial_data <- financial_data %>%
  mutate(portfolio_return = weights[1] * deannualized_yield + weights[2] * return)

# Plotting Graphs
ggplot(financial_data, aes(x = date)) +
  geom_line(aes(y = deannualized_yield, color = "Daily Net Bond Yield")) +
  geom_line(aes(y = log_returns, color = "Log Returns Bond")) +
  labs(title = "Bond Yield Analysis", x = "Date", y = "Value", color = "Series") +
  theme_minimal()

# Sharpe Ratio
sharpe_ratio <- (mean(financial_data$portfolio_return) - mean(financial_data$deannualized_yield)) /
                sd(financial_data$portfolio_return)

# Unit Root Testing and ARIMA Modeling
adf_test_price <- adf.test(financial_data$adjusted, alternative = "stationary")

adf_test_return <- adf.test(financial_data$return, alternative = "stationary")
```

### Python:

```
pip install yfinance
```
```
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Download S&P500 data
sp500_data = yf.download('^GSPC', start='2000-05-19', end='2018-06-06')

# Download Canadian 3-month treasury bills data.
bond_data = yf.download('BIL', start='2000-05-19', end='2018-06-06')

# Combine datasets
combined_data = pd.merge(bond_data, sp500_data, left_index=True, right_index=True, how='inner')

# Handle missing data
combined_data = combined_data.ffill().bfill()

# Data Preprocessing
combined_data['log_returns_bond'] = np.log(1 + combined_data['Close_x'].pct_change())
combined_data['deannualized_yield'] = combined_data['Close_x'] / 253

# Portfolio Analysis
weights = {'bond': 0.5, 'sp500': 0.5}
combined_data['portfolio_return'] = weights['bond'] * combined_data['deannualized_yield'] + \
                                     weights['sp500'] * combined_data['Close_y'].pct_change()

# Calculate mean, variance, standard deviation
mean_portfolio = combined_data['portfolio_return'].mean()
variance_portfolio = combined_data['portfolio_return'].var()
std_dev_portfolio = combined_data['portfolio_return'].std()

# Plotting Graphs
plt.plot(combined_data.index, combined_data['deannualized_yield'], label='Daily Net Bond Yield')
plt.plot(combined_data.index, combined_data['log_returns_bond'], label='Log Returns Bond')
plt.title('Bond Yield Analysis')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Sharpe Ratio
sharpe_ratio = (mean_portfolio - combined_data['deannualized_yield'].mean()) / std_dev_portfolio

# Unit Root Testing and ARIMA Modeling 
result_price = sm.tsa.adfuller(combined_data['Close_y'])
result_return = sm.tsa.adfuller(combined_data['portfolio_return'].dropna())
```

## Speech Recognition Demo

```
   pip install Flask SpeechRecognition
```
```
from flask import Flask, render_template, request
import speech_recognition as sr

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if the post request has the file part
        if "file" not in request.files:
            return render_template("index.html", error="No file part")

        file = request.files["file"]

        # If the user does not select a file, browser also
        # submit an empty part without filename
        if file.filename == "":
            return render_template("index.html", error="No selected file")

        if file:
            try:
                # Perform Speech Recognition
                recognizer = sr.Recognizer()
                audio = sr.AudioFile(file)
                with audio as source:
                    audio_data = recognizer.record(source)

                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio_data)

                return render_template("index.html", text=text)

            except sr.UnknownValueError:
                return render_template("index.html", error="Speech Recognition could not understand audio")

            except sr.RequestError as e:
                return render_template("index.html", error=f"Could not request results from Google Speech Recognition service; {e}")

    return render_template("index.html", error=None, text=None)

if __name__ == "__main__":
    app.run(debug=True)
```
#### HTML Template
```
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Speech Recognition Platform</title>
</head>
<body>
    <h1>Speech Recognition Platform</h1>
    {% if error %}
        <p style="color: red;">{{ error }}</p>
    {% endif %}
    <form method="post" enctype="multipart/form-data">
        <label for="file">Upload Audio File:</label>
        <input type="file" name="file" accept=".wav, .mp3">
        <button type="submit">Submit</button>
    </form>
    {% if text %}
        <h2>Transcription:</h2>
        <p>{{ text }}</p>
    {% endif %}
</body>
</html>

python app.py
```

## Handwriting Recognition Demo

```
pip install tensorflow
```
```
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train_flat = X_train.reshape((X_train.shape[0], -1)).astype('float32')
X_test_flat = X_test.reshape((X_test.shape[0], -1)).astype('float32')

X_train_flat /= 255.0
X_test_flat /= 255.0

y_train_onehot = to_categorical(y_train)
y_test_onehot = to_categorical(y_test)

X_train, X_val, y_train, y_val = train_test_split(X_train_flat, y_train_onehot, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, input_dim=784, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)

loss, accuracy = model.evaluate(X_test_flat, y_test_onehot)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

predictions = model.predict(X_test_flat)
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    ax.set_title(f"True: {y_test[i]}, Predicted: {np.argmax(predictions[i])}")
plt.show()
```
```
python handwriting_recognition_tf.py
```

## Sentiment Analysis Demo
This code fetches stock news articles for Amazon, AMD, and Facebook from Yahoo Finance, performs sentiment analysis using NLTK, and visualizes the sentiment scores with Matplotlib.

```
pip install nltk beautifulsoup4 matplotlib
```
```
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

# NLTK resources
nltk.download('vader_lexicon')

def get_news_sentiment(url):
    # Fetch HTML content
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract text from HTML
    paragraphs = soup.find_all('p')
    text = ' '.join([paragraph.get_text() for paragraph in paragraphs])

    # Analyze sentiment
    sia = SentimentIntensityAnalyzer()
    sentiment_score = sia.polarity_scores(text)['compound']
    return sentiment_score

# URLs for stock news articles
urls = {
    'Amazon': 'https://finance.yahoo.com/quote/AMZN/news?p=AMZN',
    'AMD': 'https://finance.yahoo.com/quote/AMD/news?p=AMD',
    'Facebook': 'https://finance.yahoo.com/quote/FB/news?p=FB'
}

# sentiment scores for each stock
sentiments = {ticker: get_news_sentiment(url) for ticker, url in urls.items()}

df = pd.DataFrame(list(sentiments.items()), columns=['Stock', 'Sentiment Score'])
df.set_index('Stock', inplace=True)

# Plot sentiment scores
df.plot(kind='bar', rot=0, color='skyblue', legend=False)
plt.title('Sentiment Analysis of Stock News')
plt.xlabel('Stocks')
plt.ylabel('Sentiment Score')
plt.show()
```
```
python sentiment_analysis.py
```

## Crime Rate Data Analysis in R

A panel data on violent crimes reported by the police in different provinces of Canada were chosen to be studied from 1998 to 2017. This analysis looks at violent crimes including homicide, attempted murder, assault, sexual assault, and robbery.

Below are some graphs depicting the trend in crime ratio over the years:

![image](https://github.com/Ayesha-Shoaib/Ayesha-Shoaib-Portfolio/assets/158636211/1d202b5a-0938-4948-a060-518b786068ae)

![image](https://github.com/Ayesha-Shoaib/Ayesha-Shoaib-Portfolio/assets/158636211/148f8c2a-e831-4a18-9e8c-bd2d76d29930)

Data descriptive analysis of the provinces is presented to explore the data before performing any regression analysis. Throughout the past 20 years, the trend in crime rate was not constant in any of the provinces, however, the level of crime is lower in 2017 than in 1998 for all of the provinces.

##### Below are descriptive statistics where:
CR is the violent crime rate in all provinces per 1,00,000 people.
UN is the rate of unemployment per 1,00,000 people.
INC is the average income in dollars each year.
PP is police presence i-e the number of police officers per 1,00,000 people.
LI is the percentage of people who live in low-income families.
YTH is the number of young people admitted to correctional services.
HS is the percentage of people enrolled in High School in the respective years.

|         CR       UN       INC       PP       LI       YTH       HS
----------------------------------------------------------------------
|Mean  | 1578.293  7.73   135649.7   184.85   10.68   6427.088    26
|Median| 1476.02   6.7    73883      188.1    10.75   4609.5      12
|Std.De| 525.6393  3.742  115765     17.14    2.6     6994.61     27.5
|Min   | 789.4     3.4    53520      143.7    5.3     0           4.03
|Max   | 3135.99   16.6   457207     218.9    16.9    32825       18.31
------------------------------------------------------------------------
#### Methodology:
Resorting to a panel approach of these statistics for provinces was to eliminate bias as much as possible as panel data gives more data variation, less collinearity and more degrees of freedom. 

We chose our independent variables based on the data available present on the provinces selected. The model specified is
CR = f(UN, INC, PP, LI, YTH, HS)

After collecting the data, there were some missing values for some years. This has been addressed by getting the average between the value in the year preceding and the following year. A regression on the variables could give us a positive coefficient suggesting a positive relationship between the regressor and crime rate. We are expecting to see a negative relationship between income level and crime rate i-e as the level of income decreases in households, the crime rate
rises whereas for people living in low-income households, we are expecting to see a positive coefficient.

The second phase of data analysis is panel regression analysis. We could perform any needed transformations on the variables e.g. if the residuals have a skewed distribution then we might need to transform the data to adjust for that. We can also plot the residuals and examine the pattern to see if it indicates heteroskedasticity, we can then perform further analysis to correct for it. We can also check for multicollinearity between the variables and correct for that
even though having panel data suggests less collinearity. To empirical test the relationship between these economic factors and the crime rate for violent crimes, we can introduce the following model:
𝐶𝑟𝑖𝑚𝑒 = α+β𝑋+ε
𝑖𝑡         𝑖𝑡  𝑖𝑡

Where α is the fixed effect, whereas β is the vector of exogenous factors discussed. ε is the error term which also includes the unobserved effects per province that may be correlated with some of the exogenous variables.

In the panel model, the individual effect terms can be modeled using two types of regression.

The fixed effect model can be estimated first to fix for the cross sections in the data using the OLS regression method. The random effect model can be estimated using the OLS method as well.

##### Fixed effect model:
In this case, intercept terms are assumed to be independent of 𝑋. The regression equation in terms of a single explanatory variable would be:

𝐶𝑟𝑖𝑚𝑒 = α+ β𝑋+ 𝑢
𝑖𝑡       𝑖𝑡  𝑖𝑡   𝑖

Where α are intercept terms that vary across provinces but remain invariant across time.

#### Random effect model:
The intercept terms may be correlated with the explanatory variables. In this case, α are modeled as random variables and are treated at par with the error term.

𝐶𝑟𝑖𝑚𝑒 = β𝑋+ ε
𝑖𝑡       𝑖𝑡   𝑖𝑡

The error terms of the same cross-sectional units can become correlated, though errors from different cross-sectional units are independent.

If the individual effects are correlated with the other regressors in the model, the fixed effect model is consistent and the random effects model is inconsistent. On the other hand, if the individual effects are not correlated with the other regressors in the model, both random and fixed effects are consistent, and random effects are efficient.

To choose whether a fixed or random model is preferred, we could use the Hausman test. It will test whether the unique errors are correlated with the regressor variables.

𝐻 : Random effect model is appropriate
0
𝐻 : Fixed effect model is appropriate
𝑎

If the probability is less than 0.05 we could reject the null hypothesis and use the fixed model, if it is more than 0.05, we have no significant evidence against the null hypothesis and use the random effect model. Then we could look at the statistical significance of the impact of regressor variables on crime rate at a 5% significance level and discuss the empirical context.

It is also essential to analyze the theoretical interpretation behind the previous empirical findings and compare our results to those.
```

library(tidyverse)
library(plm)

data <- read.csv('Smoke_data.csv')

crime_data <- as.data.frame(data)

# Impute missing values by taking the average of the previous and following year
crime_data <- crime_data %>%
  arrange(province, year) %>%
  mutate_if(is.numeric, function(x) ifelse(is.na(x), (lag(x) + lead(x)) / 2, x))

# Panel regression model
# Taking 'CR' is the dependent variable and 'UN', 'INC', 'PP', 'LI', 'YTH', 'HS' as the independent variables
model <- plm(CR ~ UN + INC + PP + LI + YTH + HS, data = crime_data, index = c("province", "year"), model = "random")

# Test for fixed or random effects using Hausman test
hausman_test <- phtest(model, alternative = "two.sided")

# Choose the appropriate model 
if (hausman_test$p.value < 0.05) {
  # Fixed effect model is appropriate
  final_model <- plm(CR ~ UN + INC + PP + LI + YTH + HS, data = crime_data, index = c("province", "year"), model = "within")
} else {
  # Random effect model is appropriate
  final_model <- plm(CR ~ UN + INC + PP + LI + YTH + HS, data = crime_data, index = c("province", "year"), model = "random")
}

summary(final_model)
```
This script performs a panel regression analysis using both fixed and random effects models and selects the appropriate model based on the Hausman test. The results and coefficients are displayed in the summary.




