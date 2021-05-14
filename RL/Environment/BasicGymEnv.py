import gym
import numpy as np
from gym import spaces
# import MarioGame

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
        self.action_space = spaces.MultiDiscrete([1, 1, 1, 1])
        # 각각 좌 / 우 / 점프 / 가속을 의미함

        # 볼 수 있는 환경을 의미함
        self.observation_space = spaces.Box(low=0, high=1, shape=(123, 123, 3), dtype=np.float)
        # 최대/최소값이 1/0으로 정규화된 3차원 numpy array를 입력으로 받음, 현재 row, col은 123인데, 게임 보고 변경해야할듯

        # Mario Game import
        # self.game = MarioGame()

    def reest(self):
        # 에피소드의 시작에 불려지며, 게임을 reset한다.
        # pyGame 모듈을 import해서 사용할 것
        # self.game.reset()

        # observation 하는 함수도 있어야 함 (pyGame 내부에서 호출할 수 있으면 좋을 듯.
        observation = 0
        # observation = self.game.get_image()   # 이미지를 return해야 함
        return observation

    def step(self, action):
        # agent의 action 결과를 받는다.
        # Multi-Discrete 환경이라서 어떻게 받는지는 모르겠지만, 일단 4개 numpy array를 받는다고 가정하자.
        action = np.max(action, 1)

        # action을 토대로 game에 입력을 줌
        # self.game.getInput(action)

        # 이후 observation을 받음
        # observation = self.game.get_image()
        # done = self.game.is_finished()

        # 받은 observation을 토대로 reward 측정
        # reward = self.reward_calculation(observation)

        # 결과값 return
        # return observation, reward, done, {"pressed":, action}

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