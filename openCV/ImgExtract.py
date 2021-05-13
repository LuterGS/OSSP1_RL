import numpy as np
import pygame.camera
import pygame.image
import cv2

"""
class ImgExtract:
    def __init__(self, pos, size):
        self.pos = pos
        self.size = size
    #cv2.imshow('window',cv2.cvtColor(screen,cv2.COLOR_BGR2GRAY))
"""

def Capture(display,pos, size):  # (pygame Surface, String, tuple, tuple)
    image = pygame.Surface(size)  # Create image surface
    image.blit(display, pos, size)  # Blit portion of the display to the image

    ar = np.array(image)
    print(ar)

def Capture2(screen,fps,Captured_fps):
    if fps % 5 == 0:
        screencopy = screen.copy()
        imgdata = pygame.surfarray.array3d(screencopy)
        CVImg = cv2.cvtColor(imgdata,cv2.COLOR_BGR2GRAY)
        CVImg = cv2.flip(CVImg,1)
        CVImg = cv2.rotate(CVImg,cv2.ROTATE_90_COUNTERCLOCKWISE)
        #CVImg = cv2.resize(CVImg,(640,480))
        cv2.imshow('window',CVImg)
        return CVImg
