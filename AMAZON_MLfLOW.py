import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from geopy.distance import great_circle
import mlflow
import mlflow.sklearn

# Load and preprocess data
df = pd.read_csv('C:/Users/SYAMNARAYANAN/OneDrive/Desktop/visual code/Guvi_project_1/Guvi_project_2/amazon_delivery (1).csv')
df['Agent_Rating'] = df['Agent_Rating'].fillna(df['Agent_Rating'].median()).astype(float)
df['Order_Time'] = df['Order_Time'].fillna('00:00:00')
df = df.dropna(subset=['Weather'])

df['Order_Date'] = pd.to_datetime(df['Order_Date'], format='%Y-%m-%d')
df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce')
df['Order_Hour'] = df['Order_Time'].dt.hour
df['Order_Minute'] = df['Order_Time'].dt.minute
df['Order_Time'] = df['Order_Time'].dt.time

df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce')
df['Pickup_Hour'] = df['Pickup_Time'].dt.hour
df['Pickup_Minute'] = df['Pickup_Time'].dt.minute
df['Pickup_Time'] = df['Pickup_Time'].dt.time

df['Pickup_Timestamp'] = df.apply(lambda row: datetime.combine(row['Order_Date'], row['Pickup_Time']), axis=1)
df['Delivered_Timestamp'] = df['Pickup_Timestamp'] + pd.to_timedelta(df['Delivery_Time'], unit='m')
df['Time_From_Pickup_To_Delivery'] = (df['Delivered_Timestamp'] - df['Pickup_Timestamp']).dt.total_seconds() / 60
df['Time_From_Pickup_To_Delivery'] = df['Time_From_Pickup_To_Delivery'].round(2)

# Clean and encode
df['Traffic'] = df['Traffic'].astype(str).str.strip()
df['Weather'] = df['Weather'].astype(str).str.strip()
Traffic_mapping = {'Jam': 1, 'High': 2, 'Medium': 3, 'Low': 4, 'Unknown': 0}
Weather_mapping = {'Sunny': 1, 'Stormy': 2, 'Cloudy': 3, 'Windy': 4, 'Fog': 5, 'Sandstorms': 6, 'Unknown': 0}
df['Traffic'] = df['Traffic'].map(Traffic_mapping)
df['Weather'] = df['Weather'].map(Weather_mapping)

vehicle_encoder = LabelEncoder()
area_encoder = LabelEncoder()
category_encoder = LabelEncoder()
df['Vehicle'] = vehicle_encoder.fit_transform(df['Vehicle'])
df['Area'] = area_encoder.fit_transform(df['Area'])
df['Category'] = category_encoder.fit_transform(df['Category'])

def calculate_geodesic_distance(row):
    store_coords = (row['Store_Latitude'], row['Store_Longitude'])
    drop_coords = (row['Drop_Latitude'], row['Drop_Longitude'])
    return great_circle(store_coords, drop_coords).km

df['Distance_km'] = df.apply(calculate_geodesic_distance, axis=1).round(1)
df['Order_Time'] = pd.to_datetime(df['Order_Time'], format='%H:%M:%S', errors='coerce')
df['Pickup_Time'] = pd.to_datetime(df['Pickup_Time'], format='%H:%M:%S', errors='coerce')
df['Total_Time_Taken'] = (df['Pickup_Time'] - df['Order_Time']).dt.total_seconds() / 60
df['Total_Time_Taken'] = df['Total_Time_Taken'].fillna(0).astype(int)

df['Distance_Traffic_Interaction'] = df['Distance_km'] * df['Traffic']
df['Time_From_Pickup_To_Delivery'] = df['Delivery_Time'] - df['Pickup_Minute']

# Features and targets
X = df[['Time_From_Pickup_To_Delivery', 'Agent_Rating', 'Weather', 'Agent_Age',
        'Traffic', 'Vehicle', 'Distance_km', 'Category', 'Total_Time_Taken']]
y = df['Delivery_Time']

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

models = [
    ("DecisionTree", DecisionTreeRegressor()),
    ("RandomForest", RandomForestRegressor()),
    ("GradientBoosting", GradientBoostingRegressor()),
    ("XGBoost", XGBRegressor()),
    ("AdaBoost", AdaBoostRegressor()),
    ("KNeighbors", KNeighborsRegressor()),
    ("SVR", SVR())
]

# MLflow tracking
mlflow.set_experiment("Amazon_Delivery_Prediction")

for name, model in models:
    with mlflow.start_run(run_name=name):
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(rmse)

        print(f"{name} R² Score: {r2:.4f}")
        print(f"{name} RMSE: {rmse:.4f}")
        print("-" * 30)

        # Log parameters and metrics
        mlflow.log_param("model", name)
        mlflow.log_metric("r2_score", r2)
        mlflow.log_metric("rmse", rmse)

        # Log model
        mlflow.sklearn.log_model(model, f"{name}_model")


