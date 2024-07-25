# save this as process_csv.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def main():
    # Load the data
    data = pd.read_csv('data.csv')
    
    # Split into features and labels
    X = data[['feature1', 'feature2', 'feature3']]
    y = data['label']
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model to a file
    output_dir = '/output'
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(output_dir, 'model.pkl'))
    print("Model saved to /output/model.pkl")

if __name__ == "__main__":
    main()

