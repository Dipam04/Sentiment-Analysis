import pandas as pd
from src.Comp.logger import logging
import os
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import save_npz


def load_data(train_url:str,test_url:str) -> pd.DataFrame:
    try:
        train_data = pd.read_csv(train_url)
        test_data = pd.read_csv(test_url)
        logging.info('Loading the processed data...')

        return train_data,test_data
    except pd.errors.ParserError as e:
        print(f'Failed to parse the csv file {e}')
        raise
    except Exception as e:
        logging.error(e)
        raise
    
def tfidf(train_data:pd.DataFrame,test_data:pd.DataFrame):
    try:
        X_train = train_data.iloc[:,-1].values
        y_train = train_data['oe__Sentiment'].values

        X_test = test_data.iloc[:,-1].values
        y_test = test_data['oe__Sentiment'].values

        tf = TfidfVectorizer(ngram_range=(1,2),
                             max_features=1500,
                             sublinear_tf=True)
        
        X_train_trf = tf.fit_transform(X_train)
        X_test_trf = tf.transform(X_test)

        return X_train_trf,X_test_trf,y_train,y_test
    
    except Exception as e:
        logging.error(f'Error in transforming the data: {e}')
        raise


def save_data(X_train,X_test,y_train,y_test,data_path:str):
    try:
        os.makedirs(data_path,exist_ok=True)

        X_train_path = os.path.join(data_path,'X_train.csv')
        X_test_path = os.path.join(data_path,'X_test.csv')

        y_train_path = os.path.join(data_path,'y_train.csv')
        y_test_path = os.path.join(data_path,'y_test.csv')

        pd.DataFrame(X_train.toarray()).to_csv(X_train_path,index=False)
        pd.DataFrame(X_test.toarray()).to_csv(X_test_path,index=False)

        pd.DataFrame(y_train,columns=['Sentiment']).to_csv(y_train_path,index=False)
        pd.DataFrame(y_test,columns=['Sentiment']).to_csv(y_test_path,index=False)

        logging.info('Saving data...')

    except Exception as e :
        logging.error(f'An error occured during saing the data {e}')
        raise

def main():
    try:
        train_data, test_data = load_data(
            train_url= r'D:\Sentimen Analysis\processed\train_processed.csv',
            test_url= r'D:\Sentimen Analysis\processed\test_processed.csv')
        
        X_train,X_test,y_train,y_test = tfidf(train_data,test_data)

        save_data(X_train,X_test,y_train,y_test,data_path='features')

    except Exception as e:
        logging.error(f'Error in main: {e}')
        raise



if __name__ == '__main__':
    main()

