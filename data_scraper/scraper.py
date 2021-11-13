from pyexpat import model
import pandas as pd
import quandl
from datetime import date
#from model.model import all_stocks

quandl.ApiConfig.api_key = "7_ZCpgsvL5Jcj9FBAAh3"

today = date.today()
d1 = today.strftime("%Y-%m-%d") #year-month-day
print("Today's date:", d1)

all_stocks = {'AAPL', 'GOOGL', 'TSLA', 'F', 'NVDA', 'VZ'}
start_dates = { 'AAPL' : "2010-01-01", 'GOOGL' : "2010-01-01", 'TSLA' : "2010-01-01", 
                'F' : "2010-01-01", 'NVDA' : "2010-01-01", 'VZ' : "2010-01-01"}
end_dates = { 'AAPL' : "2013-01-01", 'GOOGL' : "2013-01-01", 'TSLA' : "2013-01-01",
                'F' : "2013-01-01", 'NVDA' : "2013-01-01", 'VZ' : "2013-01-01"}
#company="AAPL", start="2020-01-01", end = "2021-01-01"
def scrape_data(company, start, end):
    stock_data = quandl.get("WIKI/"+ company, start_date=start, end_date=end)
    stock_data.to_csv('stock_info/'+company+'.csv')

def main():
    for company in all_stocks:
        scrape_data(company, start_dates[company], end_dates[company])
        print("Finished: " + company)
    print("Finished scraping Stock Info")


if __name__ == "__main__":
    main()