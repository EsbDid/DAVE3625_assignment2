import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 


data = pd.read_csv('/Users/esbendidriksen/Downloads/TSLA.csv')


data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values(by='Date')



X = data['Date'].values.astype(np.int64) // 10**9 
y = data['Close'].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


model = LinearRegression()
model.fit(X_train.reshape(-1, 1), y_train)


flag = True
while (flag):
    input_date_prompt = input("For predicting future price for a stock in TESLA Motors, enter a future date (YYYY-MM-DD):")

    try:
        input_date = datetime.datetime.strptime(input_date_prompt, "%Y-%m-%d")

        current_date = datetime.datetime.now()
    
        if input_date < current_date:
            print("\nPlease enter a date in the future.")
        else:
            flag = False
        
    except ValueError:
        print("\nEnter a valid date format(YYYY-MM-DD).")


        
        
input_date_datetime = input_date.timestamp()


predicted_price = model.predict([[input_date_datetime]])

MSE = mean_squared_error(y_test, model.predict(X_test.reshape(-1, 1)))

latest_date = data['Date'].max()

latest_price = data[data['Date'] == latest_date]['Close'].values[0]


percentage_score = (latest_price/predicted_price)*100


predicted_price_wsqb = str(predicted_price)[1:-1]

percentage_score_wsqb = str(percentage_score)[1:-1]


print(f"\nThe predicted price for a stock in TESLA Motors on {input_date_prompt} is ${predicted_price_wsqb:}")
print("\nPrediction percentage score:", percentage_score_wsqb,"%")
print("\nThe mean squared error:",MSE)

