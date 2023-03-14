import pandas as pd
import numpy as np
from dataclasses import dataclass
import os, sys
from src.logger import logging
from src.exception import CustomException
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessing_file_path:str = os.path.join('artifact', 'preprocessing.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessing_file(self):

        try:

            numerical_columns = ["writing_score", "reading_score"]

            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            logging.info(f'numerical columns are : {numerical_columns}')
            logging.info(f'Catergorical columns are : {categorical_columns}')

            logging.info('Creating the numerical pipeline')
            num_pipeline = Pipeline(
                [
                ('imputer',SimpleImputer(strategy="median")),
                ('scaling', StandardScaler())
                ]
            )

            logging.info('Creating the categorical pipeline')
            cat_pipeline = Pipeline(
                [
                ('imputer', SimpleImputer(strategy = 'most_frequent')),
                ('one_hot_encoder', OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Creating the column transformer object')
            column_transformer_obj = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            return column_transformer_obj
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def Initiate_Data_Transformation(self, train_path, test_path):

        try:

            preprocessing_obj = self.get_preprocessing_file()

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Imported the training and testing data')

            TARGET_COLUMN = 'math_score'
            X_train_df = train_df.drop(TARGET_COLUMN, axis = 1)
            y_train_df = train_df[TARGET_COLUMN]

            X_test_df = test_df.drop(TARGET_COLUMN, axis = 1)
            y_test_df = test_df[TARGET_COLUMN]
            logging.info('Splitted the independent and dependednt variables both from train and test data')

            X_train_df = preprocessing_obj.fit_transform(X_train_df)
            X_test_df = preprocessing_obj.transform(X_test_df)
            logging.info('Applied preprocessing techniques on both training and testing data')

            logging.info(f'{X_train_df.shape}, {y_train_df.shape}')
            train_arr = np.c_[X_train_df, np.array(y_train_df)]
            test_arr = np.c_[X_test_df, np.array(y_test_df)]

            save_object(file_path=self.data_transformation_config.preprocessing_file_path, obj=preprocessing_obj)
            logging.info('Saved the preprocessing object')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessing_file_path,
            )


        except Exception as e:
            raise CustomException(e, sys)
