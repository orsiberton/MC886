import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor


def some_function(n_loops, t):
    y = 0
    for x in range(n_loops):
        y = (t + 1) * 1000 + x
        print('process: {}, z: {}'.format(t, y))
        time.sleep(1)
    return y


def multi_threading():
    pool = ThreadPoolExecutor(max_workers=8)
    res = []
    for i in range(100):
        a = pool.submit(some_function, n_loops=10, t=i)
        res.append(a)
    for i in range(10):
        print(res[i].result())


def multi_processing():
    n_cpus = multiprocessing.cpu_count()
    pool = ProcessPoolExecutor(max_workers=n_cpus)
    res = []
    for i in range(10):
        a = pool.submit(some_function,  n_loops=10, t=i)
        res.append(a)
    for i in range(10):
        print(res[i].result())


if __name__ == '__main__':
    multi_threading()
    #multi_processing()