import pandas as pd  
from sklearn.preprocessing import StandardScaler  
import os  
import joblib
  
def preprocess_data(folder, filename):  
    data = pd.read_csv(os.path.join(folder, filename))  
    scaler = StandardScaler()  

    data['temperature'] = scaler.fit_transform(data[['temperature']])  
    processed_filename = filename.replace('.csv', '_processed.csv')  
    data.to_csv(os.path.join(folder, processed_filename), index=False)  
    
    joblib.dump(scaler, 'scaler.joblib') 

def main():  
    preprocess_data('train', 'train_data.csv')  
    preprocess_data('test', 'test_data.csv')  

if __name__ == "__main__":  
    main()