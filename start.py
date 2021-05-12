import os
import threading
import time
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

start_1 = None


def loadStart():
    global start_1
    num_img = cv2.imread(BASE_DIR + "start_1.jpg", cv2.IMREAD_GRAYSCALE)

    # 임계값보다 크면 흰색, 작으면 검은색 50~255
    _, num_img = cv2.threshold(num_img, 50, 255, cv2.THRESH_BINARY_INV)  # 이진화하기
    start_1 = num_img

# flag1 = False
# flag2 = False  # for double check
# def checkStart(img):
#     global flag1, flag2, start_1

#     # 내가 설정한 범위 고정정인 장소가 되어야함.
#     roi = img[252:335, 480:545]
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#     _, roi = cv2.threshold(roi, 50, 255, cv2.THRESH_BINARY_INV)
#     diff = cv2.bitwise_xor(roi, start_1)
#     diff_cnt = cv2.countNonZero(diff)
