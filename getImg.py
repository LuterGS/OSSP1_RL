# getImg.py

import sys
import numpy as np
import cv2
import time
import win32gui
from mss import mss
import glob
import os


win_pos = None
def getImg():
    global win_pos
    sct = mss()
    hwnd = win32gui.FindWindow(None, "Mario Client")
    if hwnd == 0:
        quit("quit")
    rect = win32gui.GetWindowRect(hwnd)
    x = rect[0] + 3 # 위치조정가능.
    y = rect[1] + 34
    win_pos = {'top': y, 'left': x, 'width': 1024, 'height': 760} # 크기는 임의로 조정가능.
    sct_img = sct.grab(win_pos)
    img = np.array(sct_img)
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

def getWinPos():
    global win_pos
    return win_pos

FPS = []
def calcFRS(prevtime):
    global FPS
    curtime = time.time()
    latency = curtime - prevtime
    try:
        fps = 1/latency
        FPS.append(fps)
    except:
        print("error")
    if len(FPS) == 40:
        print("FPS: %.0f" % (sum(FPS)/len(FPS))
        FPS = []
    return curtime

