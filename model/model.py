import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

all_stocks = {'AAPL', 'GOOGL', 'TSLA', 'F', 'NVDA', 'VZ' }

def prepare_data(df,forecast_col,forecast_out,test_size):
    label = df[forecast_col].shift(-forecast_out) #creating new column called label with the last 5 rows are nan
    X = np.array(df[[forecast_col]]) #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_lately = X[-forecast_out:] #creating the column i want to use later in the predicting method
    X = X[:-forecast_out] # X that will contain the training and testing
    label.dropna(inplace=True) #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0) #cross validation

    response = [X_train,X_test , Y_train, Y_test , X_lately]
    return response

def generate_model(X_train, X_test, Y_train, Y_test , X_lately):
    learner = LinearRegression() #initializing linear regression model
    #training the linear regression model
    learner.fit(X_train,Y_train) 
    #testing the linear regression model
    score=learner.score(X_test,Y_test)
    #set that will contain the forecasted data
    forecast= learner.predict(X_lately)
    return forecast, score

def forecast_all_stocks(forecast_col, forecast_out, test_size):
    #read in the dataset.
    for company in all_stocks:
        datafile = pd.read_csv('stock_info/'+company+'.csv', header=0, index_col='Date', parse_dates=True)
        #calling the method were the cross validation and data preperation
        X_train, X_test, Y_train, Y_test , X_lately = prepare_data(datafile,forecast_col,forecast_out,test_size); 
        forecast, score = generate_model(X_train, X_test, Y_train, Y_test, X_lately) 
        print_response(forecast, score, company)

def print_response(forecast, score, company="default"):
    #create a json output object
    response={}
    response['Company']=company
    response['test_score']=score
    response['forecast_set']=forecast
    print(response)


def main():

    forecast_col = 'Close'  #choose the column you would like to make a forcast of
    forecast_out = 5        #choose the total number of predictions
    test_size = 0.2         #choose the percentage of samples you would like to sample
    forecast_all = True

    if forecast_all is True:
        forecast_all_stocks(forecast_col, forecast_out, test_size)
    else:
        #read in the dataset.
        datafile = pd.read_csv("stock_info/AAPL.csv")
        dataset = datafile
        #calling the method were the cross validation and data preperation
        X_train, X_test, Y_train, Y_test , X_lately = prepare_data(dataset,forecast_col,forecast_out,test_size); 
        forecast, score = generate_model(X_train, X_test, Y_train, Y_test, X_lately) 
        print_response(forecast, score)
