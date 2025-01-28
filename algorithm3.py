import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

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
    
    return np.array(X), np.array(y)

# Train the model
def train_model(X, y, friction_data, material_properties, max_attempts=1000, tolerance=0.05):
    """Train the model until it achieves desired accuracy on known examples"""
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    best_model = None
    best_error = float('inf')
    
    # Different network architectures to try
    architectures = [
        (50, 25),
        (100, 50),
        (200, 100),
        (100, 50, 25),
        (200, 100, 50),
        (300, 150, 75),
        (400, 200, 100),
    ]
    
    for attempt in range(max_attempts):
        # Cycle through different architectures
        hidden_layers = architectures[attempt % len(architectures)]
        
        # Create and train the model
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation='relu',
            solver='adam',
            max_iter=2000,  # Increased max iterations
            random_state=attempt,
            learning_rate_init=0.001,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=50  # More iterations before early stopping
        )
        
        model.fit(X_train_scaled, y_train)
        
        # Check accuracy on known examples
        errors = []
        for entry in friction_data:
            if entry['staticDry'] is None:
                continue
                
            actual = process_static_dry(entry['staticDry'])
            if actual is None:
                continue
                
            material1 = entry['material1']
            material2 = entry['against_material']
            
            if material1 not in material_properties or material2 not in material_properties:
                continue
                
            predicted = predict_friction(model, scaler, material_properties, material1, material2)
            error_percentage = abs(predicted - actual) / actual
            errors.append(error_percentage)
        
        max_error_percentage = max(errors)
        avg_error_percentage = sum(errors) / len(errors)
        
        # If this is the best model so far, save it
        if max_error_percentage < best_error:
            best_error = max_error_percentage
            best_model = model
            print(f"New best model found at attempt {attempt + 1}")
            print(f"Architecture: {hidden_layers}")
            print(f"Max error: {max_error_percentage:.1%}")
            print(f"Average error: {avg_error_percentage:.1%}\n")
        
        # If we're within tolerance, we can stop
        if max_error_percentage <= tolerance:
            print(f"\nFound acceptable model after {attempt + 1} attempts")
            print(f"Final architecture: {hidden_layers}")
            print(f"Maximum error percentage: {max_error_percentage:.1%}")
            print(f"Average error percentage: {avg_error_percentage:.1%}")
            return model, scaler
    
    print(f"\nUsing best model found after {max_attempts} attempts")
    print(f"Best maximum error percentage: {best_error:.1%}")
    return best_model, scaler

# Predict friction for new materials
def predict_friction(model, scaler, material_properties, material1, material2):
    if material1 not in material_properties or material2 not in material_properties:
        return None
        
    property_names = list(material_properties[material1].keys())
    
    # Combine properties of both materials
    combined_props = []
    for prop in property_names:
        combined_props.append(material_properties[material1][prop])
        combined_props.append(material_properties[material2][prop])
    
    # Scale the input features
    combined_props_scaled = scaler.transform([combined_props])
    
    # Make prediction
    prediction = model.predict(combined_props_scaled)[0]
    return prediction

def main():
    # Load and prepare data
    material_properties, friction_data = load_data()
    X, y = prepare_dataset(material_properties, friction_data)
    
    # Train the model
    model, scaler = train_model(X, y, friction_data, material_properties)
    
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
            
        prediction = predict_friction(model, scaler, material_properties, material1, material2)
        print(f"\nPredicted static friction coefficient: {prediction:.3f}")

if __name__ == "__main__":
    main()
