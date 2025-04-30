import pandas as pd  
from sklearn.linear_model import LinearRegression  
import joblib  

def train_model():  
    data = pd.read_csv('train/train_data_processed.csv')  
    X = data.index.values.reshape(-1, 1)
    y = data['temperature'].values  

    model = LinearRegression()  
    model.fit(X, y)  
 
    joblib.dump(model, 'temperature_model.joblib')  

if __name__ == "__main__":  
    train_model()