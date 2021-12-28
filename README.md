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

The entire problem statement will be divided into 2 components and it is applicable for both the scenarios:

To address the first issue where there is a need to identify the number of cars in the images generated from the road cameras, it will be classified as a computer vision problem.

### Computer Vision

(Computer Vision) - An automated video surveillance system that uses Deep learning models to identify the object of interest in real time. Hence, for the use case it will be used to extract the number of cars from the [Traffic Condition API](https://data.gov.sg/dataset/traffic-images) at a specific time and a specific location.

The first component is composed in the following steps:

1. Change of datetime format
2. Insert datetime into API get request
3. Create image extraction module
4. Deploy Neural Network models to process the images
5. Save the output results from the model extraction

However, in this current project, machine learning life cycle is not considered. Therefore, the model will need to be update accordingly when there are more data.

The first time to manage is the date time format. There is a need to change the format where the API calls accept the datetime in the html request format.

```
variable = past_date_week.isoformat()
print(urllib.parse.quote(variable))
new_variable = urllib.parse.quote(variable)
```

Only after this step is done, users will be able to specify the date time there they would like to start the data collection process.

Subsequently, it is the detection of the image information extration module. The identification of objects, the library [cvlib](https://www.cvlib.net) was adopted to identify the cars on the road. THe library uses the open source yoloV3 model for identification.

```
import cvlib as cv

from cvlib.object_detection import draw_bbox

bbox, label, conf = cv.detect_common_objects(img)
output_image = draw_bbox(img, bbox, label, conf)
car = label.count('car')
bus = label.count('bus')
truck = label.count('truck')
motorcycle = label.count('motorcycle')
```

Each of these variables (cars, bus ...) will be stored in a panda table and be passed to the next components. However, in the purpose of this project, the number of vehicles will be the sum of all these variables and named "Total".

There are some points of considerations that are not adressed in the project:

Data collection perspective:

- The weather conditions (heavy rain, very bright aftenroon)
- Light conditions(between day time and night time)
- Taking accuracy level of the model on singapore traffic use case
- Other roads condition at the entrance of the specific location
- Refresh rate of traffic camera

The reason for highlighting these unaddress areas is that they are crutial component in the hyperparameter tuning, if another YOLOv3 model is to be applied or self-trained.

Furthermore, the modulde in wrapped into a function that will be taking in the following parameters:

```
#Changable Variables
current_date = datetime.now()
forecast_period = 3             #duration of forecast
days = 7                        #duration of the dataset collection
interval = 30                   #time between each image
latitude = 1.357098686          #location of the camera
longitude = 103.902042          #location of the camera

df = data_generation(current_date, days, interval, latitude, longitude)
#print(df)
```

Each of these variables enable users to determine the time of which they would like to forecast the number of vehicles at that particular moment. However, the approach taken here will not invole any storing of data generated, it will be pass on as a data frame onto the subsequent functions.

There are some area of improvement that can be made for the project would be improving on the stability of the API retrieval of the dataset as there are times where the module be running without and data coming in.

The data that is currently being used in this project is a

- 30 minutes interval image captured from the API server
- 7 Days of data which gives a data share of (336,1) - for the univariate analysis
- 7 Days of data which gives a data share of (336,2) - for the multivariate analysis

Overall, in the future there can be more data being use for this project. There are also some areas of condersion that are not address in the process of collection components:

- error handling (in the event where APIs are down)
- duration of data acquisition
- in the event where network is down, how can the download be resumed without restarting

### Analytics

##### Univerate Forecasting

- holiday season of singapore when we collected 1 year of data
- actual weather condition rather than weather temperature

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
