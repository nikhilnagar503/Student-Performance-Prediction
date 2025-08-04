#import libraies
import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
   
    def get_data_transformer_object(self):
        try:
           
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]
            )
            cat_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy="most_frequent")), ('one_hot_encoder',OneHotEncoder()),('scaler',StandardScaler(with_mean=False))]
            )
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            #we created the piplines and now we need to compine them by using columntransform
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipline,numerical_columns),
                ('cat_pipeline', cat_pipeline,categorical_columns)
            ])
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
         
            #now we will create the function thaat will perform the entire workflow
            #the output of data ingestion
    def initiate_data_transformation(self,train_path,test_path):
        
        try:
                 
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")
                 #creating the objoect
            preprocessing_obj=self.get_data_transformer_object()
            
            logging.info("preprocessing object created")

            target_column_name="math_score"
            numerical_columns = ["writing_score", "reading_score"]
            
            
            logging.info(f"target column name: {target_column_name}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            logging.info(f"train_df {train_df.columns}")
            logging.info(f"test_df {test_df.columns}")
            
            logging.info(f"train_df {train_df.head()}")
            logging.info(f"test_df {test_df.head()}")
            
            
                 #xtrain
            input_feature_train_df= train_df.drop(columns=[target_column_name],axis=1)
            
            logging.info(f"Input feature train dataframe: {input_feature_train_df.columns}")
            
                 #ytrain
            target_feature_train_df=train_df[target_column_name]
            
            logging.info(f"Target feature train dataframe: {target_feature_train_df.name}")

                 #xtest
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
                 #ytest
                 
            logging.info(f"Input feature test dataframe: {input_feature_test_df.columns}")
            
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Target feature test dataframe: {target_feature_test_df.name}")
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
                 #apply the prosses
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            
            train_arr = np.c_[
            input_feature_train_arr, np.array(target_feature_train_df)
            ]
            
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessing_obj)
            
            
            return (train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path)
            
            
        except Exception as e:
            raise CustomException(e, sys)
        
        