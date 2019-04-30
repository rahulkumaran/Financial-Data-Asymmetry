import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import quandl

nifty_list = pd.read_csv("nifty50.csv")
company_codes = nifty_list['Symbol'].tolist()

collective = []
#company="SBIN"

i=0

for company in company_codes:
	try:
		i+=1
		stock = "NSE/" + company
		historic_data = quandl.get(stock, api_key="DKdN7j6Q2vCNSJ_GsSf6")

		filename = stock
		historic_data.to_csv("HistoricData_" + company + ".csv")

		print(i)

	except:
		i+=1
		print(company)
		continue
	

d = pd.read_excel("HistoricData.xlsx")

#print(d)
print(size(collective))
