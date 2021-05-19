import numpy as np
import pygame.camera
import pygame.image
import cv2

def Capture(screen,fps,Captured_fps,scales):
    if fps % Captured_fps == 0:
        screencopy = screen.copy()
        imgdata = pygame.surfarray.array3d(screencopy)
        imgdata = cv2.cvtColor(imgdata,scales)
        imgdata = cv2.flip(imgdata,1)
        imgdata = cv2.rotate(imgdata,cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imshow('window',imgdata)
        return imgdata
