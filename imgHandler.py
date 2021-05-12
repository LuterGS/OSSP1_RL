from openCV.start import loadStart
from openCV.getImg import getImg


import cv2
import time
import multiprocessing
import numpy as np


def loadData():
    loadStart()


def resetData():
    global shared_dict
    shared_dict['moster']
    shared_dict['moster_attack']
    shared_dict['wall']
    shared_dict['coin']
    shared_dict['start']
    shared_dict['end']
    shared_dict['points']
    shared_dict['player_edge'] = None # 플레이어아이콘
    shared_dict['simple_map'] = 0


def getPoints():  # 특징점
    return shared_dict['points']

def getPlayerEdge():  # 플레이어 아이콘 테두리
    return shared_dict['player_edge']

def getSimpleMap():
    return shared_dict['simple_map']  # (142, 179, 3) numpy array
    # 142 -> y축, 179 -> x축, 3 -> BGR
    # (255, 255, 255) -> white, (255, 0, 0) -> blue, (0, 255, 0) -> green (0, 0, 255) -> red

"""
def shared_dict['moster']
    return shared_dict['moster']
def shared_dict['moster_attack']
    return shared_dict['moster_attack']
def shared_dict['wall']
    return shared_dict['wall']
def shared_dict['coin']
    return shared_dict['coin']
def shared_dict['start']
    return shared_dict['start']
def shared_dict['end']
    return shared_dict['end']
"""
