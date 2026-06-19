import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Load dataset
data = pd.read_csv('Bengaluru_House_Data.csv')  # Replace with your dataset's file path

# Preview the data
print(data.head())

# Feature selection and target variable
# Replace 'price' with your target column name, and list the features to include
features = ['location', 'area', 'bedrooms', 'bathrooms']  # Example features
target = 'price'

# Separate features and target
X = data[features]
y = data[target]

# Define numerical and categorical columns
numerical_features = ['area', 'bedrooms', 'bathrooms']  # Adjust as per your dataset
categorical_features = ['location']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessors in a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Define the complete pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the pipeline
model_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = model_pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Save the pipeline (optional)
import joblib
joblib.dump(model_pipeline, 'land_price_model_pipeline.pkl')

# Load the pipeline (for future use)
model_pipeline = joblib.load('land_price_model_pipeline.pkl')
