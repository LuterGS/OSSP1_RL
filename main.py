import pygame
from Pygame.classes.Dashboard import Dashboard
from Pygame.classes.MakeRandomMap import MakeRandomMap
from Pygame.classes.Level import Level
from Pygame.classes.Menu import Menu
from Pygame.classes.Sound import Sound
from Pygame.entities.Mario import Mario
from Pygame.openCV import ImgExtract
import cv2

windowSize = 640, 480
playCount = 0
clearCount = 0


def main():
    pygame.mixer.pre_init(44100, -16, 2, 4096)
    pygame.init()
    screen = pygame.display.set_mode(windowSize)
    max_frame_rate = 60
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
    print("pc",playCount)
    fps = 0
    while not mario.restart:
        if fps == 60: fps = 0
        # image Capture
        ImgExtract.Capture(screen, fps, 5, cv2.COLOR_BGR2GRAY)
        pygame.display.set_caption("Super Mario running with {:d} FPS".format(int(clock.get_fps())))
        if mario.pause:
            mario.pauseObj.update()
        else:
            level.drawLevel(mario.camera)
            dashboard.update()
            mario.update()
        pygame.display.update()
        clock.tick(max_frame_rate)
        fps += 1
    if mario.clear == True:
        clearCount += 1
        print("cc",clearCount)
    return 'restart'


if __name__ == "__main__":
    exitmessage = 'restart'
    while exitmessage == 'restart':
        exitmessage = main()
