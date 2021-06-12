import json

import gym
import numpy as np
from gym import spaces

import pygame
from Pygame.classes.Dashboard import Dashboard
from Pygame.classes.MakeRandomMap import MakeRandomMap
from Pygame.classes.Level import Level
from Pygame.classes.Menu import Menu
from Pygame.classes.Sound import Sound
from Pygame.entities.Mario import Mario
from Pygame.openCV import ImgExtract
import cv2

from multiprocessing import Process, Queue

button_log = ["left", "right", "up", "dash"]

level = "Level1-1.json"

class MultiMario(Process):
    def __init__(self, ptoc_queue: Queue, ctop_queue: Queue):
        Process.__init__(self)

        self.load_map()

        self.ptoc_queue = ptoc_queue
        # [1, 0]은 reset, [2, action]는 step으로 하자.
        self.ctop_queue = ctop_queue

        self.window_size = 640, 480
        self.play_count = 0
        self.clear_count = 0


    def load_map(self):
        with open("./Pygame/levels/Level1-1.json", 'r') as file:
            jsons = json.load(file)
        self.map_length = jsons["level"]["layers"]["ground"]["x"][1]
        self.blocks = [[float(i), 13.0] for i in range(self.map_length)]
        holes = jsons["level"]["objects"]["sky"]
        for hole in holes:
            if hole[1] == 13:
                self.blocks.pop(self.blocks.index(hole))
        pipes = jsons["level"]["objects"]["pipe"]
        for pipe in pipes:
            start, end = float(pipe[0]), float(pipe[1])
            for i in range(pipe[1], 13):
                float_i = float(i)
                self.blocks.append([start, float_i])
                self.blocks.append([end, float_i])
        additional_blocks = jsons["level"]["objects"]["ground"]
        for ad_block in additional_blocks:
            ad_block[0], ad_block[1] = float(ad_block[0]), float(ad_block[1])
            self.blocks.append(ad_block)

        # +- 10정도만 보이게

    def getEntityXY(self, mario_xy, entity_list, map_length):
        goomba_koopa = [[0.0, 0.0] for _ in range(20)]
        coins = [[0.0, 0.0] for _ in range(20)]
        stuff = [[0.0, 0.0] for _ in range(20)]
        goomba_count, koopa_count = 0, 0
        coin_count = 0
        rdbox_count, rmr_count = 0, 0

        for entity in entity_list:
            entity_pos = entity.getXY()
            if not mario_xy[0] - 10 <= entity_pos[0] <= mario_xy[0] + 10:
                continue
            if entity_pos[1] < 0:
                continue

            entity_pos[0] = entity_pos[0] / map_length
            entity_pos[1] = entity_pos[1] / 14

            if str(type(entity)) == "<class 'Pygame.entities.Goomba.Goomba'>":
                if goomba_count < 10:
                    goomba_koopa[goomba_count] = entity_pos
                    goomba_count += 1
            if str(type(entity)) == "<class 'Pygame.entities.Koopa.Koopa'>":
                if koopa_count < 10:
                    goomba_koopa[koopa_count + 10] = entity_pos
                    koopa_count += 1
            if str(type(entity)) == "<class 'Pygame.entities.Coin.Coin'>":
                if coin_count < 20:
                    coins[coin_count] = entity_pos
                    coin_count += 1
            if str(type(entity)) == "<class 'Pygame.entities.RandomBox.RandomBox'>":
                if rdbox_count < 15:
                    stuff[rdbox_count] = entity_pos
                    rdbox_count += 1
            if str(type(entity)) == "<class 'Pygame.entities.Mushroom.RedMushroom'>":
                if rmr_count < 5:
                    stuff[rmr_count + 15] = entity_pos
                    rmr_count += 1

        return np.asarray(goomba_koopa + coins + stuff)


    def observation(self, mario, entity_list):
        mario_xy = [mario.rect.x / 32, mario.rect.y / 32]

        visible_blocks = [[0.0, 0.0] for _ in range(60)]
        vb_count = 0

        for block in self.blocks:
            if not mario_xy[0] - 10 <= block[0] <= mario_xy[0] + 10:
                continue
            if vb_count >= 60:
                break

            # block[0] = block[0] / self.map_length
            # print(block[1], end="\t")
            # block[1] = block[1] / 14.0
            # print(block[1])

            visible_blocks[vb_count] = [block[0] / self.map_length, block[1] / 14.0]
            vb_count += 1
            print(visible_blocks[vb_count - 1])

        entities = self.getEntityXY(mario_xy, entity_list, self.map_length)
        # exit(0)

        res = np.asarray([visible_blocks, entities])
        # print(res)
        print(res[0][0][0])
        return np.asarray([visible_blocks, entities])

    def reset(self):

        pygame.mixer.pre_init(44100, -16, 2, 4096)
        pygame.init()
        self.screen = pygame.display.set_mode(self.window_size)
        self.max_frame_rate = 60
        self.dashboard = Dashboard("./Pygame/img/font.png", 8, self.screen)
        self.sound = Sound()
        self.level = Level(self.screen, self.sound, self.dashboard)
        self.menu = Menu(self.screen, self.dashboard, self.level, self.sound)
        self.MakeMap = MakeRandomMap()

        self.MakeMap.write_Json()

        if level == "Level1-1.json":
            self.menu.button_pressed[3] = True
            self.menu.update()

            self.menu.button_pressed[3] = True
            self.menu.update()

        while not self.menu.start:
            self.menu.update()
            self.menu.button_pressed[4] = True

        self.mario = Mario(0, 0, self.level, self.screen, self.dashboard, self.sound)
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

        self.time = self.dashboard.time
        self.pos_percent = (self.mario.rect.x / 32) / self.mario.levelObj.levelLength
        #
        # # 그 이후에 observation을 받아오고
        # observation = np.reshape(ImgExtract.Capture(self.screen, cv2.COLOR_BGR2GRAY), [480,  640, 1])

        # return 해줄 것.
        return self.observation(self.mario, self.level.returnEntityList())

    def step(self, action):

        reward = 0

        # agent의 action 결과를 받는다.
        # action의 결과를 토대로 게임의 입력값을 결정한다.
        if action[0] == action[1]:
            self.mario.input.button_pressed[0] = False
            self.mario.input.button_pressed[1] = False
            reward -= 3
        elif action[0] < action[1]:
            self.mario.input.button_pressed[0] = True
            self.mario.input.button_pressed[1] = False
            reward -= 10
        elif action[1] < action[0]:
            self.mario.input.button_pressed[0] = False
            self.mario.input.button_pressed[1] = True
            reward += 20
        self.mario.input.button_pressed[2] = False if action[2] < 0.5 else True
        self.mario.input.button_pressed[3] = False if action[3] < 0.5 else True

        # action을 토대로 game에 입력을 줌 (30FPS 기준으로 이 입력이 0.2초동안, 즉 6프레임만큼 유지된다고 가정하자
        #       -> 추후 변경 가능

        done = False

        # 입력을 기반으로 게임 진행
        for i in range(12):
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
        reward = 0

        # 게임을 클리어하지 못하고 죽었을 때
        if not self.mario.clear and done:

            # 차등 보상 적용
            pos_percent = (self.mario.rect.x / 32) / self.mario.levelObj.levelLength
            reward -= 200 * (1-pos_percent)

            # 다음값 관측
            observation = self.observation(self.mario, self.level.returnEntityList())
            return observation, reward, done, None

        # 게임을 클리어했을 때
        if self.mario.clear:
            done = True
            reward = 3000
            observation = self.observation(self.mario, self.level.returnEntityList())
            # observation = ImgExtract.Capture(self.screen, cv2.COLOR_BGR2GRAY)
            return observation, reward, done, None

        # 현재 시간과 위치를 측정
        time = self.dashboard.time
        pos_percent = (self.mario.rect.x / 32) / self.mario.levelObj.levelLength
        ypos_percent = self.mario.rect.y / 32

        # 기존의 위치랑 비교해 잘 진행했는지 비교
        mov_diff = pos_percent - self.pos_percent
        # print(pos_percent, self.pos_percent)

        # # reward 1. 거리를 토대로 더 앞으로 갔으면 +, 뒤로 갔으면 - 제공
        # if mov_diff < 0:
        #     reward -= 0.5
        # if mov_diff > 0:
        #     reward += 10000
        # if mov_diff == 0:
        #     reward -= 0.3

        # 20초를 클리어 기준으로 삼을 때
        # 넉넉하게, 4초씩 끊어서 판단함 (5개 분류, 20%)
        if 0 <= time < 4:
            pass
        elif 4 <= time < 8 and pos_percent < 20:
            reward -= 20
        elif 8 < time <= 12 and pos_percent < 40:
            reward -= 20
        elif 12 < time <= 16 and pos_percent < 60:
            reward -= 20
        elif 16 < time <= 20 and pos_percent < 80:
            reward -= 20

        self.time = time
        self.pos_percent = pos_percent

        # 이후에 observation을 받아옴
        observation = self.observation(self.mario, self.level.returnEntityList())
        # observation = np.reshape(ImgExtract.Capture(self.screen, cv2.COLOR_BGR2GRAY), [480,  640, 1])
        return observation, reward, done, pos_percent



    def program_run(self):

        while True:
            val = self.ptoc_queue.get()
            # [1, 1], [2, action]
            if val[0] == 1:
                self.ctop_queue.put(self.reset())
            if val[0] == 2:
                self.ctop_queue.put(self.step(val[1]))

    def run(self):
        self.program_run()


class BasicEnv(gym.Env):
    """Custom environment Basic code"""
    metadata = {"render.modes": ["human"]}
    action_space = spaces.MultiDiscrete([2, 2, 2, 2])
    observation_space = spaces.Box(low=0, high=1, shape=(2, 60, 2), dtype=np.float)
    reward_range = (-float(300), float(300))

    def __init__(self):
        super(BasicEnv, self).__init__()

        # Process간 통신을 위한 Queue 설정
        self.ptoc_queue = Queue()
        self.ctop_queue = Queue()

        # Mario Game Import
        self.mario = MultiMario(self.ptoc_queue, self.ctop_queue)
        self.mario.start()

    def reset(self):
        self.reset_value = 0
        self.ptoc_queue.put([1, 0])
        return self.ctop_queue.get()

    def step(self, action):

        self.reset_value += 1
        self.ptoc_queue.put([2, action])
        value = self.ctop_queue.get()
        if self.reset_value > 150:  # 20초동안 clear 못하면 reset
            value = list(value)
            value[2] = True
            value[1] -= 200 * (1-value[3])      # 못깼을때도 죽은거랑 동일한 보상 제공
            value = tuple(value)
            self.reset_value = 0
        # return 큐 값, done
        return value[0], value[1], value[2], {"location": "None"}

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
