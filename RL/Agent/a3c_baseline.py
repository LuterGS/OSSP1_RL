
import argparse
import os
from datetime import datetime
from multiprocessing import cpu_count
from threading import Thread, Lock

from multiprocessing import Process, Queue

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Flatten

from RL.Environment.BasicGymEnv import BasicEnv

import signal

realenv = BasicEnv

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="MountainCarContinuous-v0")
parser.add_argument("--num-workers", default=1, type=int)
parser.add_argument("--actor-lr", type=float, default=0.001)
parser.add_argument("--critic-lr", type=float, default=0.002)
parser.add_argument("--update-interval", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args()
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)

GLOBAL_EPISODE_NUM = 0


class Actor:
    def __init__(self, state_dim, action_dim, action_bound, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01

    def nn_model(self):
        state_input = Input(shape=(480, 640, ))
        state_input2 = Flatten()(state_input)
        dense_1 = Dense(32, activation="relu")(state_input2)
        dense_2 = Dense(32, activation="relu")(dense_1)
        out_mu = Dense(self.action_dim, activation="tanh")(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation="softplus")(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, 480, 640])
        mu, std = self.model.predict(state, use_multiprocessing=True)       # TODO : 무슨 이유에서인지 얘가 정상적으로 실행이 안됨
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(
            var * 2 * np.pi
        )
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((480, 640,)),
                Flatten(),
                Dense(32, activation="relu"),
                Dense(32, activation="relu"),
                Dense(16, activation="relu"),
                Dense(1, activation="linear"),
            ]
        )

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            # assert (
            #    v_pred.shape == td_targets.shape
            # ), f"{v_pred.shape} not equal {td_targets.shape}"

            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env_name, num_workers=cpu_count()):
        self.env_name = env_name
        self.state_dim = 2 # env.observation_space.shape
        self.action_dim = 3
        self.action_bound = 1
        self.std_bound = [1e-2, 1.0]

        self.global_actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.global_critic = Critic(self.state_dim)
        self.num_workers = num_workers


    def train(self, max_episodes=1000):
        workers = []

        for i in range(self.num_workers):
            workers.append(
                AgentThread(self.global_actor, self.global_critic, max_episodes, i)
            )

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class AgentThread(Thread):
    def __init__(self, global_actor: Actor, global_critic: Critic, max_episode, i):
        Thread.__init__(self)
        self.lock = Lock()
        self.ptoc_queue = Queue()
        self.ctop_queue = Queue()
        self.child_signal_queue = Queue()
        # 0은 글로벌 가중치 보내기 (부모 -> 자식), 1은 글로벌 가중치 받기 (자식 -> 부모), 2는 max ep 업데이트, 3은 해당 프로세스 종료

        self.global_actor = global_actor
        self.global_critic = global_critic
        self.max_ep = max_episode
        self.i = i

    def run(self):
        global GLOBAL_EPISODE_NUM
        worker = A3CWorker(self.ptoc_queue, self.ctop_queue, self.child_signal_queue, self.global_actor.model.get_weights(), self.global_critic.model.get_weights(), self.max_ep, self.i)
        worker.start()


        while True:
            value = self.child_signal_queue.get()
            if value == 0:
                self.ptoc_queue.put([self.global_actor.model.get_weights(), self.global_critic.model.get_weights()])
            if value == 1:
                weights = self.ctop_queue.get()
                with self.lock:
                    self.global_actor.model.set_weights(weights[0])
                    self.global_critic.model.set_weights(weights[1])
            if value == 2:
                GLOBAL_EPISODE_NUM += 1
                self.ptoc_queue.put(GLOBAL_EPISODE_NUM)
            if value == 3:
                break

        worker.join()



class A3CWorker(Process):
    def __init__(self, parent_to_child_queue: Queue, child_to_parent_queue: Queue, child_signal_queue: Queue, default_actor_weight, default_critic_weight, max_episodes, i):
        Process.__init__(self)
        self.env = realenv()
        self.state_dim = 2 # self.env.observation_space.shape
        self.action_dim = 3 # self.env.action_space.shape[0]
        self.action_bound = 1 # self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.max_episodes = max_episodes
        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(default_actor_weight)
        self.critic.model.set_weights(default_critic_weight)

        self.nums = i

        self.ptoc_queue = parent_to_child_queue
        self.ctop_queue = child_to_parent_queue
        self.msg_queue = child_signal_queue

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advantage(self, td_targets, baselines):
        return td_targets - baselines

    def train(self):
        with tf.device('/cpu:0'):
            cur_ep = 0
            while self.max_episodes >= cur_ep:
                state_batch = []
                action_batch = []
                reward_batch = []
                episode_reward, done = 0, False

                state = self.env.reset()
                print(f'process {self.nums} is successfully reseted, now progress...')

                while not done:
                    # self.env.render()

                    print(f'progress {self.nums} reached -1')

                    action = self.actor.get_action(state)

                    print(f'progress {self.nums} reached -0.5')

                    action = np.clip(action, -self.action_bound, self.action_bound)

                    print(f'progress {self.nums} reached 0')

                    next_state, reward, done, _ = self.env.step(action)

                    print(f'progress {self.nums} reached 0.5')


                    state = np.reshape(state, [1, 480, 640])
                    action = np.reshape(action, [1, 3])
                    next_state = np.reshape(next_state, [1, 480, 640])
                    reward = np.reshape(reward, [1, 1])
                    state_batch.append(state)
                    action_batch.append(action)
                    reward_batch.append(reward)
                    print(f'progress {self.nums} reached 1')

                    if len(state_batch) >= args.update_interval or done:
                        states = np.array([state.squeeze() for state in state_batch])
                        actions = np.array([action.squeeze() for action in action_batch])
                        rewards = np.array([reward.squeeze() for reward in reward_batch])
                        # print("in calculate (next_state) : ", next_state.shape)
                        print("shape of next_state : ", next_state.shape)
                        next_v_value = self.critic.model.predict_step(next_state)       # 원래 predict 임
                        # print("complete calculate (next_v_value) : ", next_v_value.shape)
                        td_targets = self.n_step_td_target(
                            (rewards + 8) / 8, next_v_value, done
                        )
                        print("shape of states : ", states.shape)

                        # predicts = np.zeros_like(states.shape[0])
                        # for i in range(states.shape[0]):
                        #     print("shape of states[i] : ", states[i].shape, states[i])
                        #     predicts[i] = self.critic.model.predict_step(states[i]) #, use_multiprocessing=True)

                        advantages = td_targets - self.critic.model.predict(states, use_multiprocessing=True)
                        # advantages = td_targets - predicts

                        print(f'progress {self.nums} reached 2')

                        # local weight 임시저장 - 불필요
                        # temp_actor_weight = self.actor.model.get_weights()
                        # temp_critic_weight = self.critic.model.get_weights()

                        # global 가중치 요청 및 저장
                        # signal 보내는 부분
                        print(f'process {self.nums} will request global weight')
                        self.msg_queue.put(0)
                        print(f'process {self.nums} requested global weight')

                        # 받을때까지 대기
                        print(f'process {self.nums} will get global weight')
                        global_weight = self.ptoc_queue.get()
                        print(f'process {self.nums} successfully get global weight')
                        # 받았으면 값을 현재 신경망에 적용시키고 학습
                        # 0번째가 actor, 1번째가 critic이라고 가정

                        self.actor.model.set_weights(global_weight[0])
                        self.critic.model.set_weights(global_weight[1])
                        # 이게 이제 global과 동일해짐

                        actor_loss = self.actor.train(states, actions, advantages)
                        critic_loss = self.critic.train(states, td_targets)

                        # actor_loss = self.global_actor.train(states, actions, advantages)
                        # critic_loss = self.global_critic.train(states, td_targets)
                        #
                        # self.actor.model.set_weights(self.global_actor.model.get_weights())
                        # self.critic.model.set_weights(
                        #     self.global_critic.model.get_weights()
                        # )

                        # 이제 이 신경망의 값을 다시 보냄.
                        self.msg_queue.put(1)
                        self.ctop_queue.put([self.actor.model.get_weights(), self.critic.model.get_weights()])


                        state_batch = []
                        action_batch = []
                        reward_batch = []

                    episode_reward += reward[0][0]
                    state = next_state[0]

                print(f"Episode#{cur_ep} Reward:{episode_reward} Worker:{self.nums}")
                tf.summary.scalar("episode_reward", episode_reward, step=cur_ep)

                self.msg_queue.put(2)
                cur_ep = self.ptoc_queue.get()
            self.msg_queue.put(3)

    def run(self):
        self.train()


if __name__ == "__main__":
    env_name = "Pendulum-v0"
    agent = Agent(env_name, args.num_workers)
    agent.train()
