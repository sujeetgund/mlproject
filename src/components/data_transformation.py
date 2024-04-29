import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self) -> ColumnTransformer:
        """This function is responsible for creating preprocessor object for data transformation"""
        try:
            # Features
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            # Pipelines
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            # Combining pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [("numerical_pipeline", numerical_pipeline, numerical_features),
                 ("categorical_pipeline", categorical_pipeline, categorical_features)]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            logging.info("data transformation initiated")
            # Read train and test dataset
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("train and test dataset read")

            target_feature = "math_score"
            numerical_features = ["writing_score", "reading_score"]

            logging.info("obtaining preprocess object")
            preprocessing_object = self.get_data_transformer_object()

            # Split input features and target feature from datasets
            input_features_train_df = train_df.drop(columns=[target_feature], axis=1)
            target_features_train_df = train_df[target_feature]

            input_features_test_df = test_df.drop(columns=[target_feature], axis=1)
            target_features_test_df = test_df[target_feature]

            logging.info("preprocessing train and test datasets using preprocessor object")
            input_features_train_arr = preprocessing_object.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_object.transform(input_features_test_df)

            # Concatenate columns to create train_arr and test_arr
            train_arr = np.c_[input_features_train_arr, np.array(target_features_train_df)]
            test_arr = np.c_[input_features_test_arr, np.array(target_features_test_df)]

            save_object(filepath=self.data_transformation_config.preprocessor_obj_file_path, obj=preprocessing_object)
            logging.info("saved preprocessor object")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e, sys)