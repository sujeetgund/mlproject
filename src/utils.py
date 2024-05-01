import os
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException


def save_object(filepath: str, obj) -> None:
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)

        with open(filepath, "wb") as f:
            dill.dump(obj=obj, file=f)
            f.close()
    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models: dict):
    try:
        model_report = {}

        for i in range(len(models)):
            # Get model from dictionary
            model = list(models.values())[i]

            # Train the model
            model.fit(X_train, y_train)

            # Predict test dataset
            y_test_predicted = model.predict(X_test)

            # Evaluate model
            model_score = r2_score(y_test, y_test_predicted)

            print(f"Model name: '{list(models.keys())[i]}' and Score: {model_score}")
            model_report[list(models.keys())[i]] = model_score
        
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(filepath: str):
    try:
        with open(filepath, "rb") as f:
            obj = dill.load(f)
            f.close()
            return obj
    except Exception as e:
        raise CustomException(e, sys)