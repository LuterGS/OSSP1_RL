import gym
import numpy as np
from gym import spaces

import pygame
from MarioGame.classes.Dashboard import Dashboard
from MarioGame.classes.Level import Level
from MarioGame.classes.Menu import Menu
from MarioGame.classes.Sound import Sound
from MarioGame.entities.Mario import Mario
from MarioGame.main import windowSize
from MarioGame.openCV import ImgExtract
import cv2

from MarioGame.abs_filepath import ABS_PATH

from multiprocessing import Process, Queue

button_log = ["left", "right", "up", "dash"]


class MultiMario(Process):
    def __init__(self, ptoc_queue: Queue, ctop_queue: Queue):
        Process.__init__(self)
        self.ptoc_queue = ptoc_queue
        # [1, 0]은 reset, [2, action]는 step으로 하자.
        self.ctop_queue = ctop_queue

    def reset(self):

        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        self.screen = pygame.display.set_mode(windowSize)
        self.max_frame_rate = 30
        self.dashboard = Dashboard(ABS_PATH + "img/font.png", 8, self.screen)
        self.sound = Sound()
        self.level = Level(self.screen, self.sound, self.dashboard)
        self.menu = Menu(self.screen, self.dashboard, self.level, self.sound)

        print("init complete")
        #
        # # 메뉴에서 게임 들어갈 때까지 메뉴 업데이트
        # ctr = 0
        # for i in range(2000):
        #     self.menu.update()
        # # while not self.menu.start:
        # #     ctr += 1
        # #     self.menu.update()
        # #     print(ctr)
        #
        # 1. 메뉴에서 게임 플레이로 들어감
        self.menu.key_input_status[4] = True
        self.menu.update()

        # random map 들어가는 부분
        # self.menu.key_input_status[3] = True
        # self.menu.update()
        # self.menu.key_input_status[3] = True
        # self.menu.update()

        self.menu.key_input_status[4] = True
        self.menu.update()

        # self.menu.key_input_status[4] = True
        # self.menu.update()

        # 2. 게임 플레이에서 레벨 선택
        # self.menu.key_input_status[5] = True
        # self.menu.update()

        self.mario = Mario(0, 0, self.level, self.screen, self.dashboard, self.sound)
        # mario.input.CUR_BUTTON_PRESSED 이런식으로 접근해야하는데...
        self.clock = pygame.time.Clock()

        # 게임은 프레임 단위로 나눔
        # 일단 1초간 기다리자.
        for i in range(self.max_frame_rate):
            if self.mario.pause:
                self.mario.pauseObj.update()
            else:
                self.level.drawLevel(self.mario.camera)
                self.dashboard.update()
                self.mario.update()
            pygame.display.update()
            self.clock.tick(self.max_frame_rate)

        # 그 이후에 observation을 받아오고
        observation = ImgExtract.Capture(self.screen, cv2.COLOR_BGR2GRAY)
        # print(observation)
        print("reset complete!")

        # return 해줄 것.
        return observation

    def step(self, action):
        # agent의 action 결과를 받는다.
        # Multi-Discrete 환경이라서 어떻게 받는지는 모르겠지만, 일단 4개 numpy array를 받는다고 가정하자.
        print("Action : ", action)
        button_pressed = [
            True if action[1] < action[0] else False,
            True if action[1] > action[0] else False,
            False if action[1] < 0 else True,
            False if action[2] < 0 else True
        ]
        print("Button pressed : ", button_pressed)

        # action을 토대로 game에 입력을 줌 (30FPS 기준으로 이 입력이 0.2초동안, 즉 6프레임만큼 유지된다고 가정하자
        #       -> 추후 변경 가능

        done = False
        self.mario.input.CUR_BUTTON_PRESSED = button_pressed

        # 입력을 기반으로 게임 진행
        for i in range(6):
            if self.mario.restart:
                done = True
                break

            if self.mario.pause:
                self.mario.pauseObj.update()
            else:
                self.level.drawLevel(self.mario.camera)
                self.dashboard.update()
                self.mario.update()
            pygame.display.update()
            self.clock.tick(self.max_frame_rate)

        # 이후에 observation을 받아옴
        observation = ImgExtract.Capture(self.screen, cv2.COLOR_BGR2GRAY)
        reward = -0.1  # 추후에 이미지 토대로 calculation 가능

        # reward를 일단 먼저 측정하는데, 왼쪽으로 가면 마이너스, 오른쪽으로 가면 플러스를 주자
        if button_pressed[0]:
            reward -= 1
        else:
            reward += 0.5

        if button_pressed[2]:
            reward += 0.3

        # 만약 죽었으면, 마이너스를 주자
        if done:
            reward -= 5

        # print("reward : ", reward)

        # return
        return observation, reward, done, None

        # self.game.getInput(action)

        # 이후 observation을 받음
        # observation = self.game.get_image()
        # done = self.game.is_finished()

        # 받은 observation을 토대로 reward 측정
        # reward = self.reward_calculation(observation)

        # 결과값 return
        # return [observation, reward, done, {"pressed":, action}]

    def program_run(self):

        while True:
            val = self.ptoc_queue.get()
            if val[0] == 1:
                self.ctop_queue.put(self.reset())
            if val[0] == 2:
                self.ctop_queue.put(self.step(val[1]))

    def run(self):
        self.program_run()


class BasicEnv(gym.Env):
    """Custom environment Basic code"""
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        super(BasicEnv, self).__init__()

        # 보상값을 설정함
        self.reward_range = (0, 1)
        # env.reward_range. 했을 때 출력하는 결과. reward를 조정함에 있어 이것도 변경해주는것이 좋음

        # 움직일 수 있는 방향을 의미함
        # https://github.com/LuterGS/OSSP1_RL/blob/Pygame/classes/Input.py 의 방향을 참고함.
        self.action_space = spaces.MultiDiscrete([1, 1, 1])
        # 각각 좌,우 / 점프 / 가속을 의미함

        # 볼 수 있는 환경을 의미함
        self.observation_space = spaces.Box(low=0, high=1, shape=(480, 640, ), dtype=np.float)
        # 최대/최소값이 1/0으로 정규화된 3차원 numpy array를 입력으로 받음, 현재 row, col은 123인데, 게임 보고 변경해야할듯

        # Mario Game import
        self.ptoc_queue = Queue()
        self.ctop_queue = Queue()
        self.mario = MultiMario(self.ptoc_queue, self.ctop_queue)
        self.mario.start()

    def reset(self):
        self.ptoc_queue.put([1, 0])
        return self.ctop_queue.get()

    def step(self, action):
        self.ptoc_queue.put([2, action])
        return self.ctop_queue.get()

    def reward(self, observation):
        reward = 0
        # 상의된 방식에 따라 observation에서 값을 뽑아내고, return.

        return reward

    def render(self, mode='human'):
        # 사람이 볼 수 있게끔 에이전트를 visualize 하는 부분
        # 필요 없을듯.
        pass

    def close(self):
        pass
