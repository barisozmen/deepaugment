# (C) 2019 Baris Ozmen <hbaristr@gmail.com>

"""Module consists decorators for helping other functions of the project

    Decorators are arranged into following categories:

        1. Decorator decorators:
            Those augment capabilites of other decorators
            Medics

        2. Reporters:
            Those print or log information about functions

        3. Middlemen:
            Those manipulate information entering to (arguments) and/or
            exiting from (returns) functions.

        4. Watchmen:
            Those check information entering to and exiting from functions.
            If they see an argument and/or return not obeying to rules,
            throw exception.

        5. Classy-men:
            They decorate classes fashionably

"""


from functools import wraps
import numpy as np
import pandas as pd
from importlib import reload


###########################################################
#                   Decorator decorators
###########################################################


def decorator_with_argumets(decorator):
    """Decorator of decorator allows the decorator it decorates to accept arbitrary
    number of arguments.
    
    It also allows the decorator to be used without arguments. Possible ways
    of calling the (decorated) decorator are:
        1. @decorator()
        2. @decorator(<args>)
        
    However, (decorated) decorator cannot be called without parantheses (e.g. @decorator)
    """

    def decorator_maker(*args, **kwargs):
        @wraps(decorator)
        def decorator_wrapper(func):
            return decorator(func, *args, **kwargs)

        return decorator_wrapper

    return decorator_maker


###########################################################
#                       Reporters
###########################################################


class Reporter:
    """Reporter class keeps decorators gives information about functions without intervening to them.

    They are designed to be used safely with functions without worrying about if they will
    affect the inner working of a function. They don't touch to functions, only reports.
    """

    @staticmethod
    @decorator_with_argumets
    def logger(func, logfile_dir=None):
        """Decorator logs the arguments and kwargs of the function it decorates.

        It logs to the file whose path is given by `logfile_dir` argument. If it is not given,
        it creates a new file named `<function-name>.log` in the working directory.
        """
        import logging

        reload(logging)  # ensures it works with Jupyter IPython
        # see https://stackoverflow.com/questions/18786912/

        if logfile_dir is not None:
            import os

            logging.basicConfig(
                filename=os.path.join(logfile_dir, "{}.log".format(func.__name__)),
                level=logging.INFO,
            )
        else:
            logging.basicConfig(
                filename="{}.log".format(func.__name__), level=logging.INFO
            )

        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(
                "{} ran with args: {}, and kwargs: {}".format(
                    func.__name__, args, kwargs
                )
            )
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def timer(func):
        """Decorator prints running time of the function it decorates.
        """
        import time

        @wraps(func)
        def wrapper(*args, **kwargs):
            t1 = time.time()
            result = func(*args, **kwargs)
            t2 = time.time()
            print(
                "{}()'s runtime:  {} sec.".format(func.__name__, np.round((t2 - t1), 4))
            )
            return result

        return wrapper

    @staticmethod
    def matrix_gossiper(func):
        """Gossips about the input matrices of the function it decorates.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            print(
                "{}()'s is inputted matrices with shapes:  {}, {}".format(
                    func.__name__, args[0].shape, args[1].shape
                )
            )
            return func(*args, **kwargs)

        return wrapper

    @staticmethod
    def counter(func):
        """Decorator counts and logs number of times the function it decorates has been called.
        """

        @wraps(func)
        def wrapper(*args, **kwargs):
            wrapper.count = wrapper.count + 1
            result = func(*args, **kwargs)
            print("{}() call counter:  {}".format(func.__name__, wrapper.count))
            return result

        wrapper.count = 0
        return wrapper


###########################################################
#                        Middlemen
###########################################################


def multi_element_argument(func):
    """Decorator allows the function it decorates to work with multiple element first argument.
        
    If the first argument is a multi element type (such as list, tuple, set, or
    np.array), decorated function works multiple times for each, and its returnings
    are finally returned as a list.
    
    If the first argument is a single element type, decorated function works as usual.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        ex = [1, 2, 3]
        multi_element_types = [list, tuple, set, type(np.array(ex))]
        args_copy = list(args)
        first_arg = args_copy[0]
        if type(first_arg) in multi_element_types:
            first_arg = list(first_arg)
            result = []
            for i, val in enumerate(first_arg):
                new_args = [val] + args_copy[1:]
                result.append(func(*new_args, **kwargs))
        else:
            result = func(*args, **kwargs)
        return result

    return wrapper


###########################################################
#                        Watchmen
###########################################################


def check_df_arg_nonempty(func):
    """Decorator checks if the DataFrame arguments of the decorated function is not empty

    It throws error and messages the user: if any of the argument who is a DataFrame is empty;
    or if no DataFrame was given
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        df_args = []
        for arg in args:
            if isinstance(arg, pd.DataFrame):
                df_args.append(arg)

        assert len(df_args) > 0, "No DataFrame argument is entered"

        for df_arg in df_args:
            assert len(df_arg) > 0, "One of DataFrame arguments is empty"

        rv = func(*args, **kwargs)
        return rv

    return wrapper


@decorator_with_argumets
def check_df_arg_nonempty_at(func, argorder=1):
    """Decorator checks if the argument of the decorated function  is not empty.

    It does the same thing with "check_arg_nonempty()" decorator, except it checks the argument
    whose order is given (argorder), instead of checking the first argument

    Args:
        argorder (int): order of the given argument.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):

        catch = args[argorder]
        assert catch is not None, "args[{}] shouldnot be None".format(argorder)
        assert len(catch) > 0, "args[{}] size should be larger than 0".format(argorder)

        rv = func(*args, **kwargs)
        return rv

    return wrapper


###########################################################
#                   Classy-men
###########################################################


def singleton(cls):
    """Decorator makes sure the class it is decorating wont
    be instantiated more than once

    A usage example:
        @singleton
        class AnyClass:
            ...

    If/when the class called for second time, previous (single)
    instance will be returned

    Inspiration of the decorator comes from:
    https://www.python.org/dev/peps/pep-0318/#examples

    """

    instances = {}

    @wraps(cls)
    def wrapper():
        if cls not in instances:
            instances[cls] = cls()
        return instances[cls]

    return wrapper
