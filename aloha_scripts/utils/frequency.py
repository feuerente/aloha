import time
from collections import deque
from functools import wraps


def set_frequency(freq):
    """Set frequency of the function."""

    def dec(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            start = time.time()
            result = f(*args, **kwargs)
            while (time.time() - start) < (1.0 / float(freq)):
                time.sleep(0.001)
                pass
            return result

        return wrap

    return dec


def rate_limit(frequency):
    """
    Decorator to make a function rate-limited to a specified minimum interval between calls.

    :param frequency: Maximum frequency for calls to the function.
    """

    def decorator(func):
        last_time_called = [0.0]  # List so inner wrapper func can modify it.

        @wraps(func)
        def wrapper(*args, **kwargs):
            min_interval = 1 / frequency
            elapsed = time.time() - last_time_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            last_time_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper

    return decorator
