import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
df=pd.read_csv("CO2 Emissions_Canada.csv")
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
x = cdf.iloc[:, :3]
y = cdf.iloc[:, -1]
regressor = LinearRegression()
regressor.fit(x, y)
pickle.dump(regressor, open('model.pkl','wb'))



