# reset.py

#import autoit# 응용프로그램에서 창 gui및html이 아닌 팝업을 처리하도록 하는 도구. 마우스움직히 키 입력 창제어조작
from openCV.getImg import getWinPos
import numpy as np
import time

START_X = 400 # 위치를 조정해야할듯.
START_Y = 400
RESET_X = 400
RESET_Y = 400

reset_flag = False

def checkMenu(sample_pix): # 시작버튼 위치 설정필요
    if(sample_pix[0,0] == np.array([])).all():
        print("in menu")
        startFromMenu()
        return True
    return False

def startFromMenu(): # 게임이 종료되고 다시 시작할때 메뉴화면
    global reset_flag
    win_pos = getWinPos()
    x = win_pos['left']
    y = win_pos['top']
    #autoit.mouse_click("left", x + RESET_X, y + RESET_Y, 1)
    reset_flag = True