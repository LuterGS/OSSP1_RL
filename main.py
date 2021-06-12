import pygame
from Pygame.classes.Dashboard import Dashboard
from Pygame.classes.MakeRandomMap import MakeRandomMap
from Pygame.classes.Level import Level
from Pygame.classes.Menu import Menu
from Pygame.classes.Sound import Sound
from Pygame.entities.Mario import Mario
from Pygame.openCV import ImgExtract
import cv2
import numpy as np

windowSize = 640, 480
playCount = 0
clearCount = 0
# postion_persent=0;
# time=0;

def getEntityXY(mario,entityList):
    mario_xy=np.array([mario.rect.x/32,mario.rect.y/32])
    goomba_diff=[]
    koopa_diff = []
    coin_diff = []
    RandomBox_diff=[]
    for entity in entityList:
        if str(type(entity))=="<class 'Pygame.entities.Goomba.Goomba'>":
            goomba_diff.append(tuple(entity.getXY()-mario_xy))
        if str(type(entity)) == "<class 'Pygame.entities.Koopa.Koopa'>":
            koopa_diff.append(tuple(entity.getXY()-mario_xy))
        if str(type(entity)) == "<class 'Pygame.entities.Coin.Coin'>":
            coin_diff.append(tuple(entity.getXY()-mario_xy))
        if str(type(entity)) == "<class 'Pygame.entities.RandomBox.RandomBox'>":
            RandomBox_diff.append(tuple(entity.getXY()-mario_xy))

    print(goomba_diff)
    print(koopa_diff)
    print(coin_diff)
    print(RandomBox_diff)
    return goomba_diff,koopa_diff,coin_diff,RandomBox_diff

def main():
    pygame.mixer.pre_init(44100, -16, 2, 4096)
    pygame.init()
    screen = pygame.display.set_mode(windowSize)
    max_frame_rate = 30
    dashboard = Dashboard("Pygame/img/font.png", 8, screen)
    sound = Sound()
    level = Level(screen, sound, dashboard)
    menu = Menu(screen, dashboard, level, sound)
    MakeMap = MakeRandomMap()

    MakeMap.write_Json()
    while not menu.start:
        menu.update()

    mario = Mario(0, 0, level, screen, dashboard, sound)
    clock = pygame.time.Clock()
    global playCount ,clearCount
    playCount+=1
    # print("pc",playCount)
    fps = 0
    while not mario.restart:
        if fps == 30: fps = 0
        # image Capture
        ImgExtract.Capture(screen, fps, 5, cv2.COLOR_BGR2GRAY)
        pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
        if mario.pause:
            mario.pauseObj.update()
        else:
            level.drawLevel(mario.camera)
            getEntityXY(mario, level.returnEntityList())
            dashboard.update()

            # time=dashboard.time
            # postion_persent=(mario.rect.x/32)/mario.levelObj.levelLength
            # print(postion_persent)
            # print("time :",time)
            print("x",mario.rect.x/32)
            print("y",mario.rect.y/32)
            mario.update()
        pygame.display.update()
        clock.tick(max_frame_rate)
        # print(dashboard.points)
        fps += 1
    if mario.clear == True:
        clearCount += 1
        # print("cc",clearCount)
    return 'restart'


if __name__ == "__main__":
    exitmessage = 'restart'
    while exitmessage == 'restart':
        exitmessage = main()
