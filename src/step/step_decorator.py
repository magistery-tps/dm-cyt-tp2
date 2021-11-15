from functools import wraps
import datetime as dt
import logging
import pandas as pd

def step_logger(step_fn):
    @wraps(step_fn)
    def wrapper(*args, **kwargs):
        result = step_fn(*args, **kwargs)        
        result_shape = result.shape if type(result) is pd.DataFrame else len(result)

        logging.info("{} --> {}({}) --> {}".format(args[0].shape, step_fn.__name__, kwargs, result_shape))
        return result
    return wrapper