"""testing out threading"""
from typing import List
from threading import Thread

from my_helpers import decorators


@decorators.timer
def ask_user() -> None:
    user_input = input('name: ')
    print(f'hello {user_input}')


@decorators.timer
def complex_calc() -> List[float]:
    return [x ** 2 for x in range(10_000_000)]


@decorators.timer
def main():
    ask_user()
    complex_calc()


@decorators.timer
def main_multi():
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()


thread1 = Thread(target=ask_user)
thread2 = Thread(target=complex_calc)

main()
main_multi()
