from RL.Agent import A3C
import gym


env_name = 'Pendulum-v0'

def make_env():

    return gym.make('Pendulum-v0')

a3c = A3C.A3C(gym.make(env_name), 4, 200, 0.01, 3)
a3c.train(make_env)