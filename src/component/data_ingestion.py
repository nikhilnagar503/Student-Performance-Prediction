import os 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
import datetime

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import train_test_split
from src.component.data_transformation import DataTransformationConfig
from src.component.data_transformation import DataTransformation
from src.component.model_trainer import ModelTrainigConfig
from src.component.model_trainer import ModelTrainer  

@dataclass
class DataIngestionConfig:
    training_data_path: str = os.path.join('artifacts', 'train.csv')
    testing_data_path: str = os.path.join('artifacts', 'test.csv')  
    raw_data_path: str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    
    def initiate_data_ingestion(self):
        logging.info("data ingestion start")
        
        
        try :
            df = pd.read_csv('C:\\student_performance\\Student-Performance-Prediction\\Data\stud_data.csv')
            logging.info("Dataset read as pandas dataframe")
            
            df.rename(columns={"race/ethnicity": "race_ethnicity",
                   'parental level of education':'parental_level_of_education',
                   'test preparation course':'test_preparation_course',
                   'math score':'math_score',
                   'reading score':'reading_score',
                   'writing score':'writing_score'}, inplace=True)
            
            logging.info("Renamed columns in the dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.training_data_path), exist_ok=True)  
            os.makedirs(os.path.dirname(self.ingestion_config.testing_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to artifacts folder")
            logging.info("Train test split initiated")
            
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.training_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.testing_data_path, index=False, header=True)
            logging.info("Train and test data saved to artifacts folder")
            logging.info("Data ingestion completed successfully")
            
            return (
                self.ingestion_config.training_data_path,
                self.ingestion_config.testing_data_path
    
            )   
        except Exception as e:
            raise CustomException(e, sys)
        
        


if __name__ == "__main__":
   logging.info("Starting data ingestion process")
   obj=DataIngestion()
   train_data,test_data=obj.initiate_data_ingestion()
   #now i took the train and test data so it is time for data transformation
   logging.info("Data ingestion completed successfully")
   logging.info("Starting data transformation process")
   data_transformation_obj=DataTransformation()
   train_arr,test_arr,_= data_transformation_obj.initiate_data_transformation(train_data,test_data)
   
   logging.info("Data transformation completed successfully")   
    #now we have the train and test data in the form of numpy array
   logging.info("Starting model training process")
   #now the data is ready to be trained
   model_trainer_obj = ModelTrainer()
   
   
   print(model_trainer_obj.initiate_model_trainer(train_arr,test_arr))
   
   logging.info("Model training completed successfully")
   logging.info("End of the process")
    
    
