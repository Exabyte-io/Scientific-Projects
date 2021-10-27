from typing import Any, Callable, Tuple, Type

import functools


def except_with_default_value(exceptions_to_catch: Tuple[Type[BaseException]] = (Exception,),
                              default_return: Any = None) -> Callable:
    """
    Wraps the specified function to catch the specified set of errors, returning a default value.

    Args:
        exceptions_to_catch (Tuple[Type[BaseException]]): List of exceptions that will be caught. If no error
                                                         is provided,vdefaults to Exception.
        default_return (Any): Value that will be returned if the specified errors are caught.

    """

    def decorator(fun: Callable) -> Callable:
        @functools.wraps(fun)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return fun(*args, **kwargs)
            except exceptions_to_catch:
                return default_return

        return wrapper

    return decorator
