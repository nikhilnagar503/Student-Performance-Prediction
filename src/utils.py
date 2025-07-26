import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pickle
from src.exception import CustomException



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

