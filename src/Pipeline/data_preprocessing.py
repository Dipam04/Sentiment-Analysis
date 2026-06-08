import pandas as pd
import numpy as np
import os
import sys

from src.Comp.logger import logging
from src.Comp.exception import CustomException

import re
import nltk
from nltk.stem import WordNetLemmatizer

# Read the train and test data
train_data = pd.read_csv(r'D:\Sentimen Analysis\data\train.csv')
test_data = pd.read_csv(r'D:\Sentimen Analysis\data\test.csv')

# download required nltk files
nltk.download('punkt_tab')
nltk.download('wordnet')

def lemmatizer(text):
    try:
        lemmatize = WordNetLemmatizer()
        text = text.split()
        text = [lemmatize.lemmatize(x) for x in text]
        logging.info('1st step of data preprocessing -> Lemmatizer ==> Done')

        return ''.join(text)
    
    except Exception as e:
        logging.error(f'An Error Occured {e}')
        raise


def preprocess(text):
       try:
        text = lemmatizer(text)

        text = text.lower()

        text = re.sub(r'https\S+|www\S+','URL',text)
        text = re.sub(r'\S+@\S+','EMAIL',text)
        text = re.sub(r"[$£€][\d,]+","money",text)
        text = re.sub(r'\b\d+\b','number',text)
        text = re.sub(r'\s+',' ',text).strip()

        logging.info('Text processing started..')

        return text
       except Exception as e:
        logging.error(f'An error occured during text processing {e}')
        raise

def normalize_text(df):
    try:
        if 'Text' in df.columns :
            df['Text'] = df['Text'].apply(lemmatizer)
            df['Text'] = df['Text'].apply(preprocess)
            logging.info('Preprocessing Done')
        else:
            logging.error('Column not found')
        
        logging.info('Normalizing the text...')
        return df
    except Exception as e:
        logging.error(f'An error occured during the preprocess of that data {e}')
        raise CustomException(e,sys)


def save_data(train_processed_data:pd.DataFrame,test_processed_data:pd.DataFrame,data_path:str):
    try:
        os.makedirs(data_path,exist_ok=True)
        train_processed_path = os.path.join(data_path,'train_processed.csv')
        test_processed_path = os.path.join(data_path,'test_processed.csv')

        train_processed_data.to_csv(train_processed_path,index=False)
        test_processed_data.to_csv(test_processed_path,index=False)

    except Exception as e:
        print(f'An unexpected error occured during saving the data {e}')
        logging.error(e)
        raise

def main():
    try:

        logging.info('Text processing....')
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        logging.info('Saving the processed data...')
        save_data(train_processed_data,test_processed_data,data_path='processed')

    except Exception as e:
        print(f'An error occured in main function: {e}')
        logging.error(e)
        raise


if __name__ == '__main__':
    main()