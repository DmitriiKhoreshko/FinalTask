import pandas as pd  
import joblib  
import numpy as np  
from sklearn.preprocessing import StandardScaler  

def test_model():  
    model = joblib.load('temperature_model.joblib')  
    scaler = joblib.load('scaler.joblib')
    
    data = pd.read_csv('test/test_data_processed.csv')  
    X_test = data.index.values.reshape(-1, 1)  

    predictions = model.predict(X_test)  
    predictions_inverse = scaler.inverse_transform(predictions.reshape(-1, 1)) 
    
    results = pd.DataFrame({'date': data['date'], 'predicted_temperature': predictions_inverse.flatten()})  
    print(results)  

if __name__ == "__main__":  
    test_model()