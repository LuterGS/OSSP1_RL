from multiprocessing import freeze_support

from RL.Environment.BasicGymEnv_process import BasicEnv

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env


if __name__ == "__main__":
    freeze_support()

    # check_env(BasicEnv(), skip_render_check=True)
    test_env = BasicEnv()

    env = make_vec_env(BasicEnv, n_envs=1)

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=2500)
    model.save("test2")

