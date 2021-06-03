# make a prediction with a ridge regression model on the dataset
from pandas import read_csv
from sklearn.linear_model import Ridge
# load the dataset
url = 'Mall_Customers_converted.csv'
dataframe = read_csv(url)
data = dataframe.values
X, y = data[:, :-1], data[:, -1]
# define model
model = Ridge(alpha=1.0)
# fit model
model.fit(X, y)
# define new data
row = [100,0,100,100]
# make a prediction
yhat = model.predict([row])
# summarize prediction
print('Predicted: %.3f' % yhat)
