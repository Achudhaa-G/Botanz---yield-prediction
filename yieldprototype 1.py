# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Load the dataset (assumed CSV format)
df_crop = pd.read_csv('C:/Users/System Team - Vishnu\Desktop/programmer/dataset/yield_df.csv')  # replace with your dataset path

# Explore the data (optional)
print(df_crop.head())

# Features and target variable
X_crop = df_crop.drop('crop_yield', axis=1)  # all features except target
y_crop = df_crop['crop_yield']  # target variable

# Encoding categorical features (if any) like 'crop_type'
X_crop = pd.get_dummies(X_crop, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Instantiate the Random Forest Regressor
model_crop = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model_crop.fit(X_train, y_train)

# Predict on the test set
y_pred_crop = model_crop.predict(X_test)

# Evaluate the model using Mean Absolute Error (MAE)
mae_crop = mean_absolute_error(y_test, y_pred_crop)
print(f"Mean Absolute Error for Crop Yield Forecasting: {mae_crop}")

# Optionally, evaluate the feature importance
importances_crop = model_crop.feature_importances_
print(f"Feature Importances: {importances_crop}")
