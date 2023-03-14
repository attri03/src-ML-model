from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
import os, sys
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
    save_model_path:str = os.path.join('artifact', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def Initiate_model_trainer(self, train_df, test_df):
        try:

            TARGET_COLUMN = 'math_score'
            X_train = train_df[:,:-1]
            X_test = test_df[:,:-1]
            y_train = train_df[:,-1]
            y_test = test_df[:,-1]

            logging.info('Splitting the independent and dependent variables done')
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Classifier": KNeighborsRegressor(),                    
                "XGBClassifier": XGBRegressor(),
                "CatBoosting Classifier": CatBoostRegressor(verbose=False),
                "AdaBoost Classifier": AdaBoostRegressor(),
            }

            report_to_find_best_model:dict = evaluate_model(X_train=X_train, 
                                                            X_test=X_test, 
                                                            y_train=y_train,
                                                            y_test=y_test, 
                                                            models = models
                                                            )
            logging.info('report to find best model received')
            
            best_model_score = max(sorted(report_to_find_best_model.values()))

            if best_model_score<0.6:
                raise CustomException('Accuracy too low')

            logging.info('Best model identified')

            best_model_name = list(report_to_find_best_model.keys())[
                list(report_to_find_best_model.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            save_object(
                file_path=self.model_trainer_config.save_model_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info('Model trainer and evaluator phase completed')

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
