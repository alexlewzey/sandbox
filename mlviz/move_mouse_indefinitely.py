import time

import pyautogui

start = (390., 989)
top = (323., 246.)

while True:
    pyautogui.move(600., 0., duration=3.5, _pause=False)
    pyautogui.move(-600., 0., duration=3.5)
