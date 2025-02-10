import json
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pickle
import os
import datetime
import time
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

class FrictionPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.best_error = float('inf')
        self.current_attempt = 0
        self.architectures = [
            (100, 50, 25),
            (200, 100, 50),
            (300, 150, 75),
            (400, 200, 100),
            (500, 250, 125),
        ]
        
    def load_data(self):
        with open('data_a.json', 'r') as f:
            self.material_properties = json.load(f)
        with open('data_b.json', 'r') as f:
            self.friction_data = json.load(f)['frictionData']
            
    def process_static_dry(self, value):
        if value is None:
            return None
        if isinstance(value, str) and '-' in value:
            low, high = map(float, value.split('-'))
            return (low + high) / 2
        return float(value)
        
    def engineer_features(self, props1, props2):
        engineered = []
        for key in props1.keys():
            value1 = float(props1[key])
            value2 = float(props2[key])
            
            engineered.append(value1 - value2)  # Difference
            engineered.append(value1 + value2)  # Sum
            engineered.append(value1 * value2)  # Product
            engineered.append(value1 / value2 if value2 != 0 else 0)  # Ratio
            
        return engineered
        
    def prepare_dataset(self):
        X = []
        y = []
        
        for entry in self.friction_data:
            static_dry = self.process_static_dry(entry['staticDry'])
            if static_dry is None:
                continue
                
            material1 = entry['material1']
            material2 = entry['against_material']
            
            if material1 not in self.material_properties or material2 not in self.material_properties:
                continue
                
            props1 = self.material_properties[material1]
            props2 = self.material_properties[material2]
            
            features = []
            for key in props1.keys():
                features.append(float(props1[key]))
                features.append(float(props2[key]))
            
            engineered_features = self.engineer_features(props1, props2)
            features.extend(engineered_features)
                
            X.append(features)
            y.append(static_dry)
        
        return np.array(X), np.array(y)
        
    def save_state(self, model, scaler, error, attempt):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "friction_model_saves_deepseek2"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        state = {
            'model': model,
            'scaler': scaler,
            'best_error': error,
            'current_attempt': attempt,
            'timestamp': timestamp
        }
        
        filename = f"{save_dir}/friction_model_error_{error:.3f}_attempt_{attempt}_{timestamp}.pkl"
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"\nModel saved: {filename}")
        
    def load_latest_state(self):
        save_dir = "friction_model_saves_deepseek2"
        if not os.path.exists(save_dir):
            return None
            
        files = [f for f in os.listdir(save_dir) if f.endswith('.pkl')]
        if not files:
            return None
            
        latest_file = max([os.path.join(save_dir, f) for f in files], key=os.path.getctime)
        
        with open(latest_file, 'rb') as f:
            state = pickle.load(f)
        
        print(f"\nLoaded model from: {latest_file}")
        print(f"Previous best error: {state['best_error']:.1%}")
        print(f"Resuming from attempt: {state['current_attempt']}")
        
        return state
        
    def train_model(self, X, y, max_attempts=1000, tolerance=0.05, save_interval=100):
        # Try to load previous state
        state = self.load_latest_state()
        if state is not None:
            self.model = state['model']
            self.scaler = state['scaler']
            self.best_error = state['best_error']
            self.current_attempt = state['current_attempt']
        else:
            self.current_attempt = 0
            self.best_error = float('inf')
            self.scaler = StandardScaler()
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if self.scaler is None:
            self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        try:
            for attempt in range(self.current_attempt, max_attempts):
                hidden_layers = self.architectures[attempt % len(self.architectures)]
                
                model = MLPRegressor(
                    hidden_layer_sizes=hidden_layers,
                    activation='relu',
                    solver='adam',
                    max_iter=10000,
                    random_state=attempt,
                    learning_rate_init=0.001,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=50,
                    alpha=0.0001,  # L2 regularization
                    batch_size='auto',
                    shuffle=True,
                    tol=1e-4,
                    verbose=False
                )
                
                model.fit(X_train_scaled, y_train)
                
                errors = []
                for entry in self.friction_data:
                    if entry['staticDry'] is None:
                        continue
                        
                    actual = self.process_static_dry(entry['staticDry'])
                    if actual is None:
                        continue
                        
                    material1 = entry['material1']
                    material2 = entry['against_material']
                    
                    if material1 not in self.material_properties or material2 not in self.material_properties:
                        continue
                        
                    predicted = self.predict_friction(model, material1, material2)
                    error_percentage = abs(predicted - actual) / actual
                    errors.append(error_percentage)
                
                max_error_percentage = max(errors)
                avg_error_percentage = sum(errors) / len(errors)
                
                if max_error_percentage < self.best_error:
                    self.best_error = max_error_percentage
                    self.model = model
                    print(f"\nNew best model found at attempt {attempt + 1}")
                    print(f"Architecture: {hidden_layers}")
                    print(f"Max error: {max_error_percentage:.1%}")
                    print(f"Average error: {avg_error_percentage:.1%}")
                    
                    # Save when we find a better model
                    self.save_state(model, self.scaler, self.best_error, attempt)
                
                # Periodic save
                if (attempt + 1) % save_interval == 0:
                    self.save_state(self.model, self.scaler, self.best_error, attempt)
                
                if max_error_percentage <= tolerance:
                    print(f"\nFound acceptable model after {attempt + 1} attempts")
                    print(f"Final architecture: {hidden_layers}")
                    print(f"Maximum error percentage: {max_error_percentage:.1%}")
                    print(f"Average error percentage: {avg_error_percentage:.1%}")
                    self.save_state(model, self.scaler, max_error_percentage, attempt)
                    return
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            if self.model is not None:
                self.save_state(self.model, self.scaler, self.best_error, attempt)
            
    def predict_friction(self, model, material1, material2):
        if material1 not in self.material_properties or material2 not in self.material_properties:
            return None
            
        props1 = self.material_properties[material1]
        props2 = self.material_properties[material2]
        
        features = []
        for key in props1.keys():
            features.append(float(props1[key]))
            features.append(float(props2[key]))
        
        engineered_features = self.engineer_features(props1, props2)
        features.extend(engineered_features)
        
        features_scaled = self.scaler.transform([features])
        return model.predict(features_scaled)[0]

def main():
    predictor = FrictionPredictor()
    predictor.load_data()
    X, y = predictor.prepare_dataset()
    
    print("\nStarting training session...")
    print("You can interrupt training at any time with Ctrl+C")
    print("The model will be saved periodically and when better results are found")
    
    predictor.train_model(X, y, max_attempts=2000000, tolerance=0.05, save_interval=10000)
    
    while True:
        print("\nEnter material names (or 'quit' to exit):")
        material1 = input("Material 1: ")
        if material1.lower() == 'quit':
            break
            
        material2 = input("Material 2: ")
        if material2.lower() == 'quit':
            break
            
        if material1 not in predictor.material_properties:
            print(f"Error: {material1} not found in database")
            continue
        if material2 not in predictor.material_properties:
            print(f"Error: {material2} not found in database")
            continue
            
        prediction = predictor.predict_friction(predictor.model, material1, material2)
        print(f"\nPredicted static friction coefficient: {prediction:.3f}")

if __name__ == "__main__":
    main()