import pandas as pd
import numpy as np
import os
import pickle
import json
from src.Comp.logger import logging

from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score,confusion_matrix



def load_data(X_test_url,y_test_url):
    try:
        X_test = pd.read_csv(X_test_url)
        y_test = pd.read_csv(y_test_url)
        y_test = y_test.values
        logging.info('Loading X_test and y_test for model evaluation')

        return X_test,y_test
    except Exception as e:
        logging.error(f'Error raised during loading data {e}')
        raise

def load_model(model_path:str):
    try:
        with open(model_path,'rb') as f:
            model = pickle.load(f)

        logging.info('Model load successfully...')
        return model
    except FileNotFoundError:
        logging.error(f'File not found at {model_path}')

    except Exception as e:
        logging.error(f'Error occured during loading the model {e}')
        raise

def evaluate(model,X_test,y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test,y_pred)
        precision = precision_score(y_test,y_pred,average='weighted')
        recall = recall_score(y_test,y_pred,average='weighted')
        Matrix = confusion_matrix(y_test,y_pred)

        logging.info('Evalute metrics...')
        return accuracy,precision,recall,Matrix
    except Exception as e:
        logging.error(f'An error occured {e}')
        raise

def save_metrices(accuracy,precision,recall,matrix,data_path:str):
    try:
        metrics_dict = {
            'Accuracy': float(accuracy),
            'Precision': float(precision),
            'Recall Score': float(recall),
            'Confusion Matrix': matrix.tolist()
        }

        os.makedirs(data_path,exist_ok=True)
        metric_path = os.path.join(data_path,'metrics.json')

        with open(metric_path,'w') as f:
            json.dump(metrics_dict,f,indent=4)

        logging.info(f'Metrics load at {metric_path}')
    except Exception as e:
        logging.error('An error during saving the metrices')
        raise


def main():
    try:
        X_test,y_test = load_data(
            X_test_url= r'D:\Sentimen Analysis\features\X_test.csv',
            y_test_url= r'D:\Sentimen Analysis\features\y_test.csv'
        )
        model = load_model(
            model_path= r'D:\Sentimen Analysis\model\model.pkl' 
        )
        accuracy,precision,recall,matrix = evaluate(model,X_test,y_test)
        save_metrices(accuracy,precision,recall,matrix,data_path= 'metrices')

        logging.info('Execution of the evaluation process...')


    except Exception as e:
        logging.error(f'Error occured {e}')
        raise


if __name__ == '__main__':
    main()