import math
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from ta import *

class TechnicalAnalysis:
    def __init__(self, data):
        self.data = data
        self.rsi = None
        self.mfi = None
        self.roc = None
        self.cci = None

    def ROC(self, n):
        N = self.data['Close'].diff(n)
        D = self.data['Close'].shift(n)
        self.data['ROC'] = pd.Series(N/D)
        return self.data

    def CCI(self, n):
        TP = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        self.data['CCI']= pd.Series((TP - TP.rolling(window=n, center = False).mean()) / (0.015 * TP.rolling(window=n, center=False).std()))
        return self.data

    def technical_analysis(self):
        self.mfi = momentum.money_flow_index(self.data['High'], self.data['Low'], self.data['Close'], self.data['Total Trade Quantity'],n=14)
        self.rsi = momentum.rsi(self.data['Close'])
        self.roc = self.ROC(n=126)
        self.cci = self.CCI(n=20)
        self.data['MFI'] = self.mfi
        self.data['RSI'] = self.rsi

        return self.data

    def predict_price(self):
        data = self.technical_analysis()

        data = data[np.isfinite(data['ROC'])]
        inp = data[['ROC', 'MFI', 'RSI', 'CCI', 'Turnover (Lacs)']]
        op = data[['Open']]
        print(self.data['ROC'])


        print(data)
        train_x, test_x, train_y, test_y = train_test_split(inp, op, test_size = 0.25, random_state = 999)
        gbr = GradientBoostingRegressor().fit(train_x, train_y)	#Fitting and creating a model
        rfr = RandomForestRegressor().fit(train_x, train_y)


        pred = gbr.predict(test_x)		#Predicting the answers for valdiation data


        mse = mean_squared_error(pred, test_y)	#finding the mean squared error

        '''
        rmse = math.sqrt(mse)
        for value in pred:
            value = value - rmse
        '''

        score = gbr.score(test_x, test_y)

        return pred, score, mse

if(__name__ == '__main__'):
    data = pd.read_csv("datasets/HistoricData_GRASIM.csv")
    #self.data = add_all_ta_features(self.data,"Open","High","Low","Close","Total Trade Quantity", fillna=True)
    ta_obj = TechnicalAnalysis(data)
    data = ta_obj.technical_analysis()

    prediction,score,mse = ta_obj.predict_price()

    print(prediction, score)
    print(math.sqrt(mse))
    #print(data.columns)
