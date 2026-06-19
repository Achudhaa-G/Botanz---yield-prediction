import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_land = pd.read_csv('C:/Users/System Team - Vishnu/Desktop/programmer/dataset/House Price India.csv')
#replace the csv file with dataset file

print(df_land.head())
#optional point to explore data

X_land = df_land.drop('price', axis=1) #all except target
Y_land = df_land['price'] #target variable

X_land = pd.get_dummies(X_land, drop_first= True)

X_train , X_test , Y_train , Y_test = train_test_split(X_land, Y_land, test_size=0.2, random_state=42)
#spliting the data into 80-20 ratio 

model_land = RandomForestRegressor(n_estimators=100 ,random_state=42)

model_land.fit(X_train, Y_train)
#training 

Y_pred_land = model_land.predict(X_test)
#test set preddiction 

mse_land = mean_squared_error(Y_test , Y_pred_land)
print("Mean squared error for the land price prediction: {mse_land}")

importances_land = model_land.feature_importances_
#evaluating feature importance 
print(f"feature Importances: {importances_land}")
 
 #features should be about Location , land size , land type

 