import pandas as pd
import pickle
import os
from sklearn.naive_bayes import MultinomialNB

from src.Comp.logger import logging

def load_data(X_train_url,y_train_url):
    try:
        X_train = pd.read_csv(X_train_url)
        y_train = pd.read_csv(y_train_url)
        y_train = y_train.values
        logging.info('Loading X_train and y_train for model building')

        return X_train,y_train
    except Exception as e:
        logging.error(f'Error raised during loading data {e}')
        raise

def build_model(X_train,y_train):
    try:
        mnb = MultinomialNB( alpha=0.6,fit_prior=True)
        mnb.fit(X_train,y_train)
        logging.info('Build and Trainning started...')
        return mnb
    except Exception as e:
        logging.error(f'Error occured during model building {e}')
        raise

def save_model(model,data_path:str):
    try:
        os.makedirs(data_path,exist_ok=True)
        model_path = os.path.join(data_path,'model.pkl')
        with open(model_path,'wb') as f:
            pickle.dump(model,f)
        

        logging.info(f'Model successfully saved on {model_path}')
    
    except Exception as e:
        logging.error(f'An unexpected problem occured during saved the model...{e}')
        raise

def main():
    try:
        X_train,y_train = load_data(X_train_url= r'D:\Sentimen Analysis\features\X_train.csv',
                                    y_train_url= r'D:\Sentimen Analysis\features\y_train.csv')
        model = build_model(X_train,y_train)
        save_model(model,data_path = 'model')

        logging.info('Model trained successfully....')
    except Exception as e:
        logging.error(f'An error occured during trainning of the data {e}')
        raise

if __name__ == '__main__':
    main()
