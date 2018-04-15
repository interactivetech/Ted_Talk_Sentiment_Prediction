import time


def check_time(start=None, s=None):
    if start:
        print("Time for {}: {}".format(s, time.time() - start))
    else:
        return time.time()
