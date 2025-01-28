import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import tensorflow as tf
import json

# Load data_A (objects with physical properties)
with open('data_a.json', 'r') as file:
    data_A_raw = json.load(file)

# Convert data_A to DataFrame
# Convert data_A to DataFrame
data_A = pd.DataFrame([
    {
        'Material': material,
        **{k.replace(' ', '_'): v for k, v in properties.items()}
    }
    for material, properties in data_A_raw.items()
])

print("Sample of data_A:")
print(data_A.head())
print(f"Number of unique materials in data_A: {data_A['Material'].nunique()}")

# Load data_B (experiments with Static_Coefficient_Friction values)
with open('data_b.json', 'r') as file:
    data_B_raw = json.load(file)

def create_row(material_1, material_2, friction, surface_type):
    row = {
        'Material_1': material_1,
        'Material_2': material_2,
        'Static_Coefficient_Friction': friction,
        'Surface_Type': surface_type
    }
    # Add all properties for Material_1
    for prop in data_A.columns:
        if prop != 'Material':
            row[f'{prop}_1'] = data_A.loc[data_A['Material'] == material_1, prop].values[0]
    # Add all properties for Material_2
    for prop in data_A.columns:
        if prop != 'Material':
            row[f'{prop}_2'] = data_A.loc[data_A['Material'] == material_2, prop].values[0]
    return row

# Convert data_B to DataFrame
data_B = []
for item in data_B_raw['frictionData']:
    material_1 = item['material1']
    material_2 = item['against_material']
    
    if material_1 in data_A['Material'].values and material_2 in data_A['Material'].values:
        dry_friction = item['staticDry']
        lubricated_friction = item['staticLubricated']
        
        if isinstance(dry_friction, str) and '-' in dry_friction:
            dry_friction = dry_friction.split('-')[0].strip()
        if isinstance(lubricated_friction, str) and '-' in lubricated_friction:
            lubricated_friction = lubricated_friction.split('-')[0].strip()
        
        if dry_friction and dry_friction != '-':
            row = create_row(material_1, material_2, float(dry_friction), 'dry')
            data_B.append(row)
        
        if lubricated_friction and lubricated_friction not in ['-', 'null']:
            row = create_row(material_1, material_2, float(lubricated_friction), 'lubricated')
            data_B.append(row)
    else:
        print(f"Skipping entry due to missing material data: {material_1} or {material_2}")

data_B = pd.DataFrame(data_B)

print("\nSample of processed data_B:")
print(data_B.head())
print(f"Number of valid entries in data_B: {len(data_B)}")

# Check if we have valid data
if data_B.empty:
    print("No valid data found in data_B. Please check your input data.")
    exit()

# Separate dry and lubricated data
data_B_dry = data_B[data_B['Surface_Type'] == 'dry'].copy()
data_B_lubricated = data_B[data_B['Surface_Type'] == 'lubricated'].copy()

# Feature engineering function
def engineer_features(row):
    engineered_features = {}
    for prop in data_A.columns:
        if prop != 'Material':
            engineered_features[f'{prop}_diff'] = row[f'{prop}_1'] - row[f'{prop}_2']
            engineered_features[f'{prop}_sum'] = row[f'{prop}_1'] + row[f'{prop}_2']
            engineered_features[f'{prop}_product'] = row[f'{prop}_1'] * row[f'{prop}_2']
    return engineered_features

# Process dry data
X_B_dry = data_B_dry.drop(['Material_1', 'Material_2', 'Static_Coefficient_Friction', 'Surface_Type'], axis=1)
X_B_dry_engineered = data_B_dry.apply(engineer_features, axis=1, result_type='expand')
X_B_dry = pd.concat([X_B_dry, X_B_dry_engineered], axis=1)
y_B_dry = data_B_dry['Static_Coefficient_Friction']

# Process lubricated data
X_B_lubricated = data_B_lubricated.drop(['Material_1', 'Material_2', 'Static_Coefficient_Friction', 'Surface_Type'], axis=1)
X_B_lubricated_engineered = data_B_lubricated.apply(engineer_features, axis=1, result_type='expand')
X_B_lubricated = pd.concat([X_B_lubricated, X_B_lubricated_engineered], axis=1)
y_B_lubricated = data_B_lubricated['Static_Coefficient_Friction']

# Preprocess data
scaler_dry = StandardScaler()
X_B_dry_scaled = scaler_dry.fit_transform(X_B_dry)

scaler_lubricated = StandardScaler()
X_B_lubricated_scaled = scaler_lubricated.fit_transform(X_B_lubricated)

# Split the data
X_train_dry, X_test_dry, y_train_dry, y_test_dry = train_test_split(X_B_dry_scaled, y_B_dry, test_size=0.2, random_state=42)
X_train_lubricated, X_test_lubricated, y_train_lubricated, y_test_lubricated = train_test_split(X_B_lubricated_scaled, y_B_lubricated, test_size=0.2, random_state=42)

# Define and compile the neural network model
def create_nn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Create models for dry and lubricated surfaces
models_dry = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Neural Network': create_nn_model(X_train_dry.shape[1])
}

models_lubricated = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Neural Network': create_nn_model(X_train_lubricated.shape[1])
}

# Train and evaluate models for dry surfaces
print("Models for dry surfaces:")
for name, model in models_dry.items():
    if name != 'Neural Network':
        scores = cross_val_score(model, X_train_dry, y_train_dry, cv=5, scoring='neg_mean_absolute_error')
        print(f"{name} CV MAE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        model.fit(X_train_dry, y_train_dry)
    else:
        history = model.fit(X_train_dry, y_train_dry, epochs=100, validation_split=0.2, verbose=0)
        val_loss = history.history['val_loss'][-1]
        print(f"Neural Network Validation MSE: {val_loss:.4f}")

# Train and evaluate models for lubricated surfaces
print("\nModels for lubricated surfaces:")
for name, model in models_lubricated.items():
    if name != 'Neural Network':
        scores = cross_val_score(model, X_train_lubricated, y_train_lubricated, cv=5, scoring='neg_mean_absolute_error')
        print(f"{name} CV MAE: {-scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        model.fit(X_train_lubricated, y_train_lubricated)
    else:
        history = model.fit(X_train_lubricated, y_train_lubricated, epochs=100, validation_split=0.2, verbose=0)
        val_loss = history.history['val_loss'][-1]
        print(f"Neural Network Validation MSE: {val_loss:.4f}")

# Feature importance for Random Forest (dry surfaces)
rf_model_dry = models_dry['Random Forest']
feature_importance_dry = rf_model_dry.feature_importances_
for feature, importance in zip(X_B_dry.columns, feature_importance_dry):
    print(f"{feature}: {importance:.4f}")

# Feature importance for Random Forest (lubricated surfaces)
rf_model_lubricated = models_lubricated['Random Forest']
feature_importance_lubricated = rf_model_lubricated.feature_importances_
for feature, importance in zip(X_B_lubricated.columns, feature_importance_lubricated):
    print(f"{feature}: {importance:.4f}")

# Prediction function
def predict_friction(material_1, material_2, surface_type):
    if material_1 not in data_A['Material'].values or material_2 not in data_A['Material'].values:
        raise ValueError(f"One or both materials not found in the database.")
    if surface_type not in ['dry', 'lubricated']:
        raise ValueError("Surface type must be 'dry' or 'lubricated'.")
    
    # Create input data
    input_data = {}
    for prop in data_A.columns:
        if prop != 'Material':
            input_data[f'{prop}_1'] = data_A.loc[data_A['Material'] == material_1, prop].values[0]
            input_data[f'{prop}_2'] = data_A.loc[data_A['Material'] == material_2, prop].values[0]
    
    input_df = pd.DataFrame([input_data])
    engineered_features = engineer_features(input_df.iloc[0])
    input_df = pd.concat([input_df, pd.DataFrame([engineered_features])], axis=1)
    
    if surface_type == 'dry':
        X_new = input_df[X_B_dry.columns]
        X_new_scaled = scaler_dry.transform(X_new)
        return rf_model_dry.predict(X_new_scaled)[0]
    else:
        X_new = input_df[X_B_lubricated.columns]
        X_new_scaled = scaler_lubricated.transform(X_new)
        return rf_model_lubricated.predict(X_new_scaled)[0]

# Example usage with user input
try:
    material_1 = input("Enter the first material: ")
    material_2 = input("Enter the second material: ")
    surface_type = input("Enter the surface type (dry/lubricated): ").lower()
    predicted_friction = predict_friction(material_1, material_2, surface_type)
    print(f"Predicted {surface_type} friction coefficient between {material_1} and {material_2}: {predicted_friction:.4f}")
except Exception as e:
    print(f"Error in prediction: {e}")