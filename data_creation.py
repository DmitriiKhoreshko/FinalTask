import numpy as np  
import pandas as pd  
import os  

def create_temperature_data(num_days=30, include_anomalies=False):  
    np.random.seed(42)
    temperatures = np.random.normal(loc=20, scale=5, size=num_days)
    dates = pd.date_range(start='2023-01-01', periods=num_days)  

    if include_anomalies:  
        for _ in range(3):
            idx = np.random.randint(0, num_days)  
            temperatures[idx] += np.random.choice([-15, 15])

    return pd.DataFrame({'date': dates, 'temperature': temperatures})  

def save_data(data, folder, filename):  
    if not os.path.exists(folder):  
        os.makedirs(folder)  
    data.to_csv(os.path.join(folder, filename), index=False)  

def main():  
    train_data = create_temperature_data(num_days=120, include_anomalies=True)  
    test_data = create_temperature_data(num_days=30, include_anomalies=False)  

    save_data(train_data, 'train', 'train_data.csv')  
    save_data(test_data, 'test', 'test_data.csv')  

if __name__ == "__main__":  
    main()