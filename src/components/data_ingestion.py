from dataclasses import dataclass
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.logger import logging
from src.exception import CustomException

TRAIN_FILE_PATH = 'train.csv'
TEST_FILE_PATH = 'test.csv'
RAW_FILE_PATH = 'raw.csv'

@dataclass
class DataIngestionConfig:
    train_file_path:str = os.path.join('artifact', TRAIN_FILE_PATH)
    test_file_path:str = os.path.join('artifact', TEST_FILE_PATH)
    raw_file_path:str = os.path.join('artifact', RAW_FILE_PATH)
    test_size = 0.2

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def Initiate_Data_Ingestion(self):
        try:

            logging.info('Initiated data ingestion phase')
            df = pd.read_csv('src.csv')
            logging.info('read the csv file')

            arti_folder = os.path.dirname(self.data_ingestion_config.raw_file_path)
            os.makedirs(arti_folder)
            logging.info('Made the artifact folder')

            df.to_csv(self.data_ingestion_config.raw_file_path)
            logging.info('raw data saved')

            train_df, test_df = train_test_split(df, test_size = self.data_ingestion_config.test_size, random_state = 42)
            logging.info('splitted data into training and testing')

            train_df.to_csv(self.data_ingestion_config.train_file_path)
            logging.info('training data saved')

            test_df.to_csv(self.data_ingestion_config.test_file_path)
            logging.info('testing data saved')
            logging.info('Data Ingestion phase completed')
            return (
                self.data_ingestion_config.raw_file_path,
                self.data_ingestion_config.train_file_path,
                self.data_ingestion_config.test_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

