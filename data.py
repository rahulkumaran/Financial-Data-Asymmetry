import pandas as pd
import quandl

nifty_list = pd.read_csv("nifty50.csv")
company_codes = nifty_list['Symbol'].tolist()

collective = []
company="SBIN"

#for company in company_codes:
stock = "NSE/" + company
historic_data = quandl.get(stock, api_key="DKdN7j6Q2vCNSJ_GsSf6")

filename = stock
historic_data.to_excel("HistoricData.xlsx", sheet_name=company)

print(historic_data)

#print(company_codes)
