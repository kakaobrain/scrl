from functools import wraps
import logging

from distributed import comm


class ExceptionLogger(object):
    """A function decorator that helps you catch exceptions in a multiprocessing 
    environment and find out from which process they were thrown.
    """
    def __init__(self, logger_name):
        self.logger_name = logger_name
        
    def __call__(self, function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                logging.getLogger(self.logger_name).error(
                    f"The exception is occured in " 
                    f"process rank: {comm.get_rank()}"
                )
                logging.getLogger(self.logger_name).error(
                    exc, exc_info=True)
                raise exc
        return wrapped
