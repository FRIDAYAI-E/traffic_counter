# traffic_counter

Please find below the instructions for the technical assessment:

Using publicly available data at data.gov.sg please create a model pipeline to forecast (upcoming 3 hours) the traffic flow at the specified location (latitude: 1.357098686 longitude: 103.902042), for a specified time of day. The solution must have the following components:

Estimation of historical traffic flow, using image data sets available here (https://data.gov.sg/dataset/traffic-images )
Use the pipeline from (1) and weather data (https://data.gov.sg/dataset/realtime-weather-readings), to forecast the traffic flow at the specified location at a specified time of day
Notes: You may use any additional data sources / APIs to build your models

Instructions:

The model pipeline should be implemented and documented in Jupyter notebook
Your approach should also be explained on Jupyter notebook
Your evaluation approach and results should be recorded in Jupyter notebook and presented.
Please submit the Github link to your code by email, by December 21st , 6:00 PM Singapore time. No changes to the codebase are permitted thereafter

## Solving Methodology

The entire problem will be divided into 2 components for both scenarios:

- (Computer Vision) Extracting information from the APIs (images)
- (Analytics) Analytics conducted on the number of identified cars from the computer vision component

### Steps of delivery

#### Computer Vision

The very first component what was build was the functions that enable us to select

```
#Changable Variables
current_date = datetime.now()
forecast_period = 3
days = 7
interval = 30
latitude = 1.357098686
longitude = 103.902042

df = data_generation(current_date, days, interval, latitude, longitude)
#print(df)
```

Each of these variables enable users to determine the time of which they would like to forecast the number of vehicles at that particular moment. However, the approach taken here will not invole any storing of data generated, it will be pass on as a data frame onto the subsequent functions.

As for the identification of objects, the library [cvlib](https://www.cvlib.net) was adopted to identify the cars on the road. THe library uses the open source yoloV3 model for identification. Throughout the process of identifying the number of vehicles, there are some points of considerations that are not made in the project:

- the weather conditions [heavy rain]
- light conditions[between day time and night time]
- taking into account the accuracy level of the model within this use case is not tested.

There are some area of improvement that can be made for the project would be improving on the stability of the API retrieval of the dataset as there are times where the module be running without and data coming in.

The data that is currently being used in this project is a

- 30 minutes interval image captured from the API server
- 7 Days of data which gives a data share of (168,1) - for the univariate analysis
- 7 Days of data which gives a data share of (168,2) - for the multivariate analysis

Overall, in the future there can be more data being use for this project if we are able to add more data sources such as

- cross junction cameras image
- holiday season of singapore when we collected 1 year of data
- actual weather condition rather than weather temperature

This will give us a stronger dataset.

#### Analytics

##### Univerate Forecasting

In the project, 3 models were conducted to forecast the number of vehicles on the road for the next 3 hours and the models used are:

- ARIMA
- LSTM
- Prophet

Under the initial EDA of the dataset, we identified a seasonal pattern within each day of a 24 "30 mins" time period. Its more having peaks in the morning and evening period of the day.Thus we can safely say that seasonality do exist in the dataset. In addition, from the ADF test, it shows a false results which means that our data is stationary and no differencing is required to be conducted.

However from the results of the analysis, the best model with the lowest error is ARIMA as it gives the results closet to the actual test data.

#### Multivariate Forecasting

Under this model, the only model used is VAR model. There are more other models that can be explored expecially deep learning models however, it should be used in the case where more data points are captured given the current scenario of 168, it is not useful.

An initial attempt was to create new features with the intention to seve out the underlying variable that is correlated to the increase in the number of vehicles the actual hour of the day and also giving in the considerations, it will be affected by the time of the year due to holiday period.

```
#Creating new features from the weather components
df_3 = df_weather[['Date', 'Total', 'Temp']]
df_3.Date = pd.to_datetime(df_3.Date)
# day
df_3['Day'] = df_3['Date'].dt.day
# month
df_3['Month'] = df_3['Date'].dt.month
# year
df_3['Year'] = df_3['Date'].dt.year
hour
df_3['Hour'] = df_3['Date'].dt.hour
# Monday is 0 and Sunday is 6
df_3['Weekday'] = df_3['Date'].dt.weekday
# week of the year
df_3['WeekNo'] = df_3['Date'].dt.week

df_3 = df_3.set_index("Date")
```

However, in the case in order to have a year of data, it is not possible and hence, part of the features were removed. In this analysis, only temperature and total number of vehicles where used in the forecast.

There are other methods available such as the following:

- LSTM
- XGBOOST
- and many more

But due to the limitation of data, the best approach would be to deploy VAR model. There are more considers to be made.
