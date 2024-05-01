import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_models, save_object


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("model trainer initiated")

            logging.info("split input train and test data")
            X_train, y_train, X_test, y_test = (train_arr[:, :-1], train_arr[:, -1], test_arr[:, :-1], test_arr[:, -1])

            # Dictionary of models
            models = {
                "RandomForest": RandomForestRegressor(),
                "DecisionTree": DecisionTreeRegressor(),
                "GradientBoosting": GradientBoostingRegressor(),
                "LinearRegression": LinearRegression(),
                "K-NeighboursRegression": KNeighborsRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegression": CatBoostRegressor(),
                "AdaBoostRegression": AdaBoostRegressor()
            }

            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

            # # Best model score
            # best_model_score = max(model_report.values())
            # # Best model name
            # best_model_name = model_report.keys()[models.values().index(best_model_score)]

            # Best model from ChatGPT
            best_model_name = max(model_report, key=lambda k: model_report[k])
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            print(f"Best model: {best_model_name} and score is {best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found!", sys)
            logging.info("best model found on both training and testing dataset")

            # Save best model object
            save_object(filepath=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Predict
            y_predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, y_predicted)

            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)