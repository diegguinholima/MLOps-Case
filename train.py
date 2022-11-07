import pandas as pd
import numpy as np
import tensorflow as tf
import mlflow
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import Callback
from mlflow import log_metrics

class MlFlowCallback(Callback):
    """
    Implement a Custom Callback to log metrics to MLflow
    """

    def on_epoch_end(self, epoch, logs=None):
        log_metrics(logs, step=epoch)

def train_model(x_train, y_train, parameters):
    """
    Train a neural network model using the MLFlow to log the metrics 

    :param x_train: training data
    :param y_train: training labels
    """
    with mlflow.start_run():

        mlflow.tensorflow.autolog()
        model = Sequential()
        model.add(Dense(20, input_dim = x_train.shape[1], activation = 'relu')) 
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(300, activation = 'relu'))
        model.add(Dense(1000, activation = 'relu'))
        model.add(Dense(100, activation = 'relu'))
        model.add(Dense(20, activation = 'softmax'))
        model.compile(loss = parameters["loss_function"], optimizer = parameters["optimizer"], metrics = [parameters["metrics"]])
        
        model.fit(x_train, y_train, epochs=50, batch_size=10, callbacks=[MlFlowCallback()])

def preprocess(file_path: str):
    """
    Preprocess the data and return the train and test data
    The main goal of this function is similate the data preprocessing steps

    :param file_path: path to the csv file
    :return: train and test data
    """
    raw_data = pd.read_csv(file_path, delimiter=";", encoding='latin1')
    df = raw_data.drop(columns=['id', 'safra_abertura', 'data'])
    df['sexo'] = df['sexo'].replace(['F','M'],[0,1])
    df[' valor '] = df[' valor '].str.replace('-','0')
    df[' valor '] = df[' valor '].apply(lambda x: float(x.split()[0].replace('.', '').replace(',', '.')))
    df = df.dropna()
    
    Y = df.pop("grupo_estabelecimento")
    Y = pd.get_dummies(Y)
    X = df[['idade', 'sexo', ' valor ', 'limite_total', 'limite_disp']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    x_train = x_train.values
    y_train = y_train.values
    x_test = x_test.values
    y_test = y_test.values  

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    file_name = 'parameters.yml'
    with open(file_name) as fh:
        parameters = yaml.load(fh, Loader=yaml.FullLoader)
    x_train, y_train, x_test, y_test = preprocess("data/MiBolsillo.csv")
    train_model(x_train, y_train, parameters)