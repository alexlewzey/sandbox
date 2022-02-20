"""automate the mouse to rotate a 3d plotly figure at a constant speed"""
import time

import pyautogui

start = (390., 989)
top = (323., 246.)

pyautogui.position()
pyautogui.click()
pyautogui.moveTo(*start)
pyautogui.drag(0., -50., duration=0.5, button='left')
pyautogui.moveTo(*top)
for _ in range(4):
    pyautogui.drag(600., 0., duration=2.5, button='left', _pause=False)
    pyautogui.move(-600., 0., _pause=False)
