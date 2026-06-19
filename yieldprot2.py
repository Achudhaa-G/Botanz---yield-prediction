import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load your dataset
data = pd.read_csv('C:/Users/System Team - Vishnu/Desktop/CSP project codings/FAOSTAT_Yield.xls')  # Replace with your dataset file

# Inspect the dataset
print("Initial Dataset:")
print(data.head())

# Handle missing values
data.fillna(data.mean(), inplace=True)  # Replace missing values with the column mean

# Remove duplicates
data.drop_duplicates(inplace=True)

# Convert categorical data to numeric using one-hot encoding
categorical_columns = ['Soil_Type', 'Crop_Type']  # Replace with your categorical columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# Normalize numeric features (optional for Random Forest)
numeric_columns = ['Temperature', 'Rainfall']  # Replace with your numeric columns
scaler = MinMaxScaler()
data[numeric_columns] = scaler.fit_transform(data[numeric_columns])

# Inspect the cleaned dataset
print("\nCleaned Dataset:")
print(data.head())

# Visualize correlations (optional)
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Define features (X) and target (y)
# Assuming the target column is named 'Yield'
X = data.drop('Yield', axis=1)
y = data['Yield']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Feature Importance
feature_importances = pd.DataFrame(
    {'Feature': X.columns, 'Importance': rf_model.feature_importances_}
).sort_values(by='Importance', ascending=False)
print("\nFeature Importances:")
print(feature_importances)

# Save the cleaned dataset (optional)
data.to_csv('cleaned_crop_data.csv', index=False)
print("\nCleaned dataset saved as 'cleaned_crop_data.csv'")

# Save the trained model (optional)
joblib.dump(rf_model, 'crop_yield_model.pkl')
print("Trained model saved as 'crop_yield_model.pkl'")

# To load the model later:
# loaded_model = joblib.load('crop_yield_model.pkl')
