import sys
import os 
import pandas as pd
from src.Comp.logger import logging
from src.Comp.exception import CustomException

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder


def load_data(data_url: str) -> pd.DataFrame:
    logging.info('Entered the data ingestion method')
    try:
        df = pd.read_csv(data_url)
        logging.info('Read the data set as dataframe')
        return df
    except pd.errors.ParserError as e :
        print('Error: Failed to parse the csv file!!')
        print(e)
        raise
    except Exception as e:
        print('An unexpected error occured!!')
        print(e)
        raise

def data_clean(df:pd.DataFrame,data_path:str):
    logging.info('Cleaning of data nadd saving the final dataframe')
    try:
        df.drop(columns = ['ID','Entity'],inplace = True)

        preprocess = ColumnTransformer([
            ('oe',OrdinalEncoder(),['Sentiment'])
        ],remainder='passthrough')

        final_df = pd.DataFrame(preprocess.fit_transform(df),
                                columns=preprocess.get_feature_names_out())
        logging.info('Final Dataframe made after cleaning')

        os.makedirs(data_path,exist_ok=True)
        save_path = os.path.join(data_path,'Final_df.csv')
        final_df.to_csv(save_path,index=False)
        logging.info(f'Final dataframe saved to {save_path}')

        return final_df
    
    except Exception as e:
        print(f'Error during data cleaning {e}')
        logging.error(e)
        raise


def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str):
    logging.info('Saving the data.....')
    try:
        os.makedirs(data_path,exist_ok=True)
        train_path = os.path.join(data_path,'train.csv')
        test_path = os.path.join(data_path,'test.csv')

        train_data.to_csv(train_path,index=False)
        test_data.to_csv(test_path,index=False)

        logging.info(f'Train data saved to {train_path}')
        logging.info(f'Test data saved to {test_path}')

    except Exception as e:
        print(f'Error: An unexpected error occured while saving the data {e}')
        logging.error(e)
        raise


def main():
    logging.info('Execution Started....')

    try:
        df = load_data(data_url='twitter.csv')
        final_df = data_clean(df,data_path='data')
        train_data,test_data = train_test_split(final_df,test_size=0.2,random_state=42)
        save_data(train_data,test_data,data_path='data')

    except Exception as e:
        print(f'Error: {e}')
        print('Failed to complete data ingestion process')
        

if __name__ == '__main__':
    main()