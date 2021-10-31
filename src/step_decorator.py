from functools import wraps
import datetime as dt
import logging

def step_logger(step_fn):
    @wraps(step_fn)
    def wrapper(*args, **kwargs):
        result = step_fn(*args, **kwargs)
        logging.info("{} --> {}({}) --> {}".format(args[0].shape, step_fn.__name__, kwargs, result.shape))
        return result
    return wrapper