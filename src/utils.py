import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
            
        return file_path
    except Exception as e:
        raise CustomException(e, sys)
    


def load_object(file_path):
    """
    Load an object from a file using pickle.
    """
    try:
        with open(file_path, 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        
        model_report = {}
        
        for i in range(len(list(models))):
            model = list(models.keys())[i]
            
            para = params[list(models)[i]]
            
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            
            
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)
            
            model_report[list(models.keys())[i]] = test_model_score
            
            
            
        return model_report
    except Exception as e:
        raise CustomException(e, sys)
    