import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            num_features = ['ApplicantIncome',	'CoapplicantIncome',	'LoanAmount',	'Loan_Amount_Term', 'Credit_History']
            cat_features =  ['Gender','Married','Dependents','Education','Self_Employed','Property_Area',] 

        

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler()),

                ]
            )

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy= 'most_frequent')),
                    ("one_hot_encoder",OneHotEncoder()),
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")

            logging.info("categorical columns encoding completed")


            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipelines",cat_pipeline,cat_features)
                ]
            )

            return preprocessor
        

        
        except Exception as e:
            raise CustomException(e,sys)
        

    def initiate_data_transformation(self,train_path,test_path):
        

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("The train and test completed")

            # Check the shape and columns of the data
            logging.info(f"Train Data Columns: {train_df.columns}")
            logging.info(f"Test Data Columns: {test_df.columns}")
            logging.info(f"Train Data Shape: {train_df.shape}")
            logging.info(f"Test Data Shape: {test_df.shape}")


            logging.info("obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Loan_Status"
            num_features = ['ApplicantIncome',	'CoapplicantIncome',	'LoanAmount',	'Loan_Amount_Term', 'Credit_History']

            input_feature_train_df = train_df.drop(columns = [target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f"Train features shape: {input_feature_train_df.shape}")
            logging.info(f"Test features shape: {input_feature_test_df.shape}")

        
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        
            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe"
            )


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"saved preprocessing object.")


            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj 
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)