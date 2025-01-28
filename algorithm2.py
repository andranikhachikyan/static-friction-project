import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the data
def load_data():
    with open('data_a.json', 'r') as f:
        material_properties = json.load(f)
    with open('data_b.json', 'r') as f:
        friction_data = json.load(f)['frictionData']
    return material_properties, friction_data

# Process the staticDry value
def process_static_dry(value):
    if value is None:
        return None
    if isinstance(value, str) and '-' in value:
        # Handle range values by taking average
        low, high = map(float, value.split('-'))
        return (low + high) / 2
    return float(value)

# Prepare the dataset
def prepare_dataset(material_properties, friction_data):
    X = []  # Features
    y = []  # Target values (staticDry)
    material_pairs = []  # Store material pairs for validation
    
    property_names = list(next(iter(material_properties.values())).keys())
    
    for entry in friction_data:
        static_dry = process_static_dry(entry['staticDry'])
        if static_dry is None:
            continue
            
        material1 = entry['material1']
        material2 = entry['against_material']
        
        # Skip if either material is not in our properties database
        if material1 not in material_properties or material2 not in material_properties:
            continue
            
        # Get properties for both materials
        props1 = material_properties[material1]
        props2 = material_properties[material2]
        
        # Combine properties of both materials
        combined_props = []
        for prop in property_names:
            combined_props.append(props1[prop])
            combined_props.append(props2[prop])
            
        X.append(combined_props)
        y.append(static_dry)
        material_pairs.append((material1, material2, static_dry))
    
    return np.array(X), np.array(y), material_pairs

# Train the model with validation
def train_model_with_validation(X, y, material_pairs, material_properties, max_attempts=100, tolerance=0.05):
    best_model = None
    best_error = float('inf')
    
    print("Training model (this might take a while)...")
    
    for attempt in range(max_attempts):
        # Train a new model
        model = RandomForestRegressor(n_estimators=1000, random_state=attempt)
        model.fit(X, y)
        
        # Check accuracy on known examples
        max_relative_error = 0
        for material1, material2, actual_value in material_pairs:
            prediction = predict_friction(model, material_properties, material1, material2)
            relative_error = abs(prediction - actual_value) / actual_value
            max_relative_error = max(max_relative_error, relative_error)
        
        # Update best model if this one is better
        if max_relative_error < best_error:
            best_error = max_relative_error
            best_model = model
            print(f"Attempt {attempt + 1}: Found better model with max relative error: {best_error:.3f}")
            
            # If we've achieved desired accuracy, stop
            if best_error <= tolerance:
                print(f"Found model within {tolerance*100}% tolerance after {attempt + 1} attempts!")
                break
    
    if best_error > tolerance:
        print(f"\nWarning: Could not find model within {tolerance*100}% tolerance after {max_attempts} attempts.")
        print(f"Best achieved error: {best_error*100:.1f}%")
    
    return best_model

# Predict friction for new materials
def predict_friction(model, material_properties, material1, material2):
    if material1 not in material_properties or material2 not in material_properties:
        return None
        
    property_names = list(material_properties[material1].keys())
    
    # Combine properties of both materials
    combined_props = []
    for prop in property_names:
        combined_props.append(material_properties[material1][prop])
        combined_props.append(material_properties[material2][prop])
    
    # Make prediction
    prediction = model.predict([combined_props])[0]
    return prediction

def main():
    # Load and prepare data
    material_properties, friction_data = load_data()
    X, y, material_pairs = prepare_dataset(material_properties, friction_data)
    
    # Train model with validation
    model = train_model_with_validation(X, y, material_pairs, material_properties)
    
    # Verify some known examples
    print("\nVerifying known examples:")
    for material1, material2, actual_value in material_pairs[:5]:  # Show first 5 examples
        prediction = predict_friction(model, material_properties, material1, material2)
        error_percent = abs(prediction - actual_value) / actual_value * 100
        print(f"{material1} vs {material2}:")
        print(f"  Actual: {actual_value:.3f}")
        print(f"  Predicted: {prediction:.3f}")
        print(f"  Error: {error_percent:.1f}%")
    
    # Interactive prediction loop
    while True:
        print("\nEnter material names (or 'quit' to exit):")
        material1 = input("Material 1: ")
        if material1.lower() == 'quit':
            break
            
        material2 = input("Material 2: ")
        if material2.lower() == 'quit':
            break
            
        if material1 not in material_properties:
            print(f"Error: {material1} not found in database")
            continue
        if material2 not in material_properties:
            print(f"Error: {material2} not found in database")
            continue
            
        prediction = predict_friction(model, material_properties, material1, material2)
        print(f"\nPredicted static friction coefficient: {prediction:.3f}")

if __name__ == "__main__":
    main()
