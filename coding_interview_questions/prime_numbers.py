import math
from typing import List, Generator



def is_factor(x: int, y: int) -> bool:
    """return true is x in a factor of y"""
    return True if (x % y == 0) and (x != y) else False


def get_primes(n: int) -> List:
    """return a list of prime numbers in the range [2,n]"""
    primes = []
    for no in range(2, n + 1):

        sub_rng = range(2, no + 1)
        if any([is_factor(no, sub_n) for sub_n in sub_rng]):
            continue
        else:
            primes.append(no)

    return primes


def prime_gen() -> Generator[int, None, None]:
    """generator that yields successive prime numbers"""
    n = 1
    while True:
        n += 1
        sub_rng = range(2, n + 1)
        if any([is_factor(n, sub_n) for sub_n in sub_rng]):
            continue
        else:
            yield n


def if_prime_v1(n: int) -> bool:
    """return a bool indicating whether n is a prime number"""
    if n < 2:
        raise ValueError('Expected n larger than 1')

    for no in range(2, n):
        if n % no == 0:
            return False
    return True


def if_prime_v3(n: int) -> bool:
    """return a bool indicating whether n is a prime number"""
    if n < 2:
        raise ValueError('Expected n larger than 1')

    if n == 2:
        return True
    elif n % 2 == 0:
        return False

    n_sqrt = math.floor(math.sqrt(n))
    for no in range(2, n_sqrt + 1):
        if n % no == 0:
            return False
    return True


[if_prime_v1(i) for i in range(2, 100_000)]
[if_prime_v3(i) for i in range(2, 100_000)]

