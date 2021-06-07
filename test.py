from multiprocessing import freeze_support

from RL.Agent.a3c_discrete import A3C as dA3C
from RL.Agent.a3c_baseline_thread import A3C as cA3C
from RL.Environment import BasicGymEnv_process

def discrete():
    a3c = dA3C(BasicGymEnv_process.BasicEnv)
    a3c.train()

def continuous():
    a3c = cA3C(BasicGymEnv_process.BasicEnv, 6)
    # a3c.load_model("new")
    a3c.train(max_episodes=1000)

if __name__ == "__main__":
    freeze_support()
    continuous()
