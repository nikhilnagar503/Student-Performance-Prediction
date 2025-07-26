import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
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

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models


@dataclass
class ModelTrainigConfig:
    model_file_path: str = os.path.join('artifacts', 'model.pkl')
    
    
class ModelTrainer:
    def __init__(self):
        self.model_training_config = ModelTrainigConfig()
    
    def initiate_model_trainer(self, train_array, test_array):
        try:
          
          logging.info("split the training and test input")\
              
          X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
          
          
          models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
          
          logging.info("models defines successfully ")
          logging.info(f"models: {models.keys()}")
          
          params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
          
          logging.info("created parmas ")
          
          
          model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)
          
          logging.info("evaluated models")
          logging.info(f"model report: {model_report}")
          
          
          best_model_score = max(sorted(model_report.values()))
          
          logging.info(f"best model score: {best_model_score}")
          
          best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
          ]
          
          
          logging.info(f"best model name")
          best_model = models[best_model_name]
          
          logging.info(f"best model: {best_model}")
          
          
          if best_model_score<0.6:
                raise CustomException("No best model found")
          logging.info(f"Best found model on both training and testing dataset")
          
          save_object(
                file_path=self.model_training_config.model_file_path,
                obj=best_model
            )
          
          predicted=best_model.predict(X_test)
          
          r2_square = r2_score(y_test, predicted)
          return r2_square,best_model
      
        except Exception as e:
            raise CustomException(e, sys)
