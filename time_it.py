# Time code function
def time_it(func):
    import time
    import numpy as np
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        time_elapsed = time.time()-start
        print(f'Time: {np.round(time_elapsed, 2)} seconds')
        return time_elapsed, results
    return wrapper