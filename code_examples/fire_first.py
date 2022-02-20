"""
$ python fire_first.py --name=moje --n 50
"""
import fire


def greet(name, n=1):
    for _ in range(n):
        print(f'hello {name}')


if __name__ == '__main__':
    fire.Fire(greet)
