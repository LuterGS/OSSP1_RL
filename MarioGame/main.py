import pygame
from MarioGame.classes.Dashboard import Dashboard
from MarioGame.classes.Level import Level
from MarioGame.classes.Menu import Menu
from MarioGame.classes.Sound import Sound
from MarioGame.entities.Mario import Mario
from MarioGame.openCV import ImgExtract
import cv2

from MarioGame.abs_filepath import ABS_PATH

windowSize = 640, 480


def main():
    pygame.mixer.pre_init(44100, -16, 2, 4096)
    pygame.init()
    screen = pygame.display.set_mode(windowSize)
    max_frame_rate = 60
    dashboard = Dashboard(ABS_PATH + "img/font.png", 8, screen)
    sound = Sound()
    level = Level(screen, sound, dashboard)
    menu = Menu(screen, dashboard, level, sound)

    while not menu.start:
        menu.update()

    mario = Mario(0, 0, level, screen, dashboard, sound)
    clock = pygame.time.Clock()

    fps= 0
    while not mario.restart:
        if fps == 60 : fps = 0
        #image Capture
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
    return 'restart'

def start_game():
    exitmessage = 'restart'
    while exitmessage == 'restart':
        exitmessage = main()

if __name__ == "__main__":
    exitmessage = 'restart'
    while exitmessage == 'restart':
        exitmessage = main()
