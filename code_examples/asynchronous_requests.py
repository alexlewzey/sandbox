"""creating a timer"""
import concurrent
import random
import time
from concurrent.futures.thread import ThreadPoolExecutor
from threading import Thread
from typing import List, Union, Tuple
import numpy as np

stop_threads: bool = False


def fmt_seconds(secs: int) -> str:
    """format seconds human readable format hours:mins:seconds"""
    secs_per_hour: int = 3600
    secs_per_min: int = 60
    hours, remainder = divmod(secs, secs_per_hour)
    mins, seconds = divmod(remainder, secs_per_min)
    return f'{int(hours):02}:{int(mins):02}:{int(seconds):02}'


def terminal_timer(refresh_period: float = 1) -> None:
    """periodically print time elapsed to the terminal"""
    start = time.time()
    global stop_threads
    while not stop_threads:
        secs_elapsed = int(time.time() - start)
        print(f'time elapsed: {fmt_seconds(secs_elapsed)}')
        time.sleep(refresh_period)


def server_request(query: str):
    print(f'starting: {query}')
    time.sleep(random.randint(5, 8))
    print(f'completed: {query}')
    return np.random.rand(10)


def async_sql_queries(queries: Union[List, Tuple], max_workers: int = 3, refresh_period: int = 1):
    """make and return the output of asynchronous sql requests with timer"""
    global stop_threads
    data: List = []
    thread_timer = Thread(target=terminal_timer, args=(refresh_period,))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = [executor.submit(server_request, query) for query in queries]
        thread_timer.start()
        for f in concurrent.futures.as_completed(results):
            data.append(f.result())
        stop_threads = True
    return data


queries_sql = ['a', 'b', 'c', 'd']
query_data = async_sql_queries(queries_sql, max_workers=2)
print(query_data)
