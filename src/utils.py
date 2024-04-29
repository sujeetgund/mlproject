import os
import sys

import pandas as pd
import numpy as np
import dill

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