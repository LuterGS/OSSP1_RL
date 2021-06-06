
import argparse
import os
from datetime import datetime
from multiprocessing import cpu_count
from threading import Thread, Lock

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Lambda, Flatten
import tensorflow.keras.layers as layers

from RL.Environment.BasicGymEnv_process import BasicEnv

FILE_LOC = os.path.dirname(os.path.abspath(__file__))
print(FILE_LOC)

realenv = BasicEnv

tf.keras.backend.set_floatx("float64")

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="TEST")
parser.add_argument("--num-workers", default=6, type=int)
parser.add_argument("--actor-lr", type=float, default=0.01)
parser.add_argument("--critic-lr", type=float, default=0.02)
parser.add_argument("--update-interval", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--logdir", default="logs")

args = parser.parse_args()
logdir = os.path.join(
    args.logdir, parser.prog, args.env, datetime.now().strftime("%Y%m%d-%H%M%S")
)
print(f"Saving training logs to:{logdir}")
writer = tf.summary.create_file_writer(logdir)

GLOBAL_EPISODE_NUM = 0
locks = Lock()


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
        inputs = layers.Input((480, 640, 1,))
        cnn_1 = layers.Conv2D(32, 3, strides=(3, 3), activation='relu')(inputs)
        batch_norm = layers.BatchNormalization()(cnn_1)
        dropout = layers.Dropout(0.2)(batch_norm)
        flatten = layers.Flatten()(dropout)
        out_mu = layers.Dense(self.action_dim, activation='sigmoid')(flatten)
        mu_output = layers.Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = layers.Dense(self.action_dim, activation='relu')(flatten)
        return tf.keras.models.Model(inputs, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, 480, 640])
        mu, std = self.model.predict(state)
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
        return tf.keras.Sequential([
            layers.InputLayer(input_shape=(480, 640, )),
            layers.Reshape((480, 640, 1, )),
            layers.Conv2D(64, 10, strides=(10, 10,), activation='relu'),
            layers.Dropout(0.2),
            # layers.Conv2D(32, 3, activation='relu'),
            # layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

        # return tf.keras.Sequential(
        #     [
        #         Input((480, 640,)),
        #         Flatten(),
        #         Dense(32, activation="relu"),
        #         Dense(32, activation="relu"),
        #         Dense(16, activation="relu"),
        #         Dense(1, activation="linear"),
        #     ]
        # )

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

class A3C:
    def __init__(self, env_func, num_workers):
        self.state_dim = 2 # 게임의 이미지 크기
        self.action_dim = 4     # 4개
        self.action_bound = 10
        self.std_bound = [1e-2, 1.0]

        self.global_actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.global_critic = Critic(self.state_dim)
        self.num_workers = num_workers
        self.env_func = env_func

    def train(self, max_episodes=10000):
        workers = []

        for i in range(self.num_workers):
            workers.append(
                A3CWorker(self.env_func, self.global_actor, self.global_critic, max_episodes, i)
            )

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        self.save_model()
        print("Done Training")

    def save_model(self):
        print("reached Here!")
        date = datetime.now().strftime("%Y%M%d%H%M")
        actor = self.global_actor.model.get_weights()
        critic = self.global_critic.model.get_weights()
        np.save("./weights/" + date + "_actor", actor, allow_pickle=True)
        np.save("./weights/" + date + "_critic", critic, allow_pickle=True)

    def load_model(self, date_str):
        actor = np.load("./weights/" + date_str + "_actor.npy", allow_pickle=True)
        critic = np.load("./weights/" + date_str + "_critic.npy", allow_pickle=True)
        self.global_actor.model.set_weights(actor)
        self.global_critic.model.set_weights(critic)

class A3CWorker(Thread):
    def __init__(self, envs, global_actor, global_critic, max_episodes, i):
        Thread.__init__(self)
        self.env = envs()
        self.state_dim = 2
        self.action_dim = 4
        self.action_bound = 1
        self.std_bound = [1e-2, 1.0]

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(
            self.state_dim, self.action_dim, self.action_bound, self.std_bound
        )
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

        self.nums = i


    def save_model(self, number):
        print("reached Here!")
        date = datetime.now().strftime("%Y%M%d%H%M")
        actor = self.global_actor.model.get_weights()
        critic = self.global_critic.model.get_weights()
        np.save("./weights/" + date + "_" + str(number) + "_actor", actor, allow_pickle=True)
        np.save("./weights/" + date + "_" + str(number) + "_critic", critic, allow_pickle=True)

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
        global GLOBAL_EPISODE_NUM
        # print("finally reached here!")
        while self.max_episodes >= GLOBAL_EPISODE_NUM:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            state = self.env.reset()

            while not done:
                # 현재 상태로부터 취할 행동을 Actor로부터 얻어음
                action = self.actor.get_action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)

                # 얻어온 행동을 실제 Env에 작용, 다음 상태와 reward를 받아옴
                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, 480, 640])
                action = np.reshape(action, [1, 4])
                next_state = np.reshape(next_state, [1, 480, 640])
                reward = np.reshape(reward, [1, 1])
                # 각 상태, 행동, 보상을 저장함
                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    states = np.array([state.squeeze() for state in state_batch])
                    actions = np.array([action.squeeze() for action in action_batch])
                    rewards = np.array([reward.squeeze() for reward in reward_batch])
                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        (rewards + 8) / 8, next_v_value, done
                    )
                    advantages = td_targets - self.critic.model.predict(states)

                    locks.acquire()
                    actor_loss = self.global_actor.train(states, actions, advantages)
                    critic_loss = self.global_critic.train(states, td_targets)

                    self.actor.model.set_weights(self.global_actor.model.get_weights())
                    self.critic.model.set_weights(
                        self.global_critic.model.get_weights()
                    )

                    state_batch.clear()
                    action_batch.clear()
                    reward_batch.clear()
                    locks.release()

                episode_reward += reward[0][0]
                state = next_state[0]

            print(f"Episode#{GLOBAL_EPISODE_NUM} on Thread:{self.nums},  Reward:{episode_reward}")
            tf.summary.scalar("episode_reward", episode_reward, step=GLOBAL_EPISODE_NUM)
            locks.acquire()
            GLOBAL_EPISODE_NUM += 1
            if GLOBAL_EPISODE_NUM % 20 == 0:
                self.save_model(GLOBAL_EPISODE_NUM)
            locks.release()
        print(f"Thread:{self.nums} finished")

    def run(self):
        self.train()

if __name__ == "__main__":
    env_name = "Pendulum-v0"
    agent = A3C(env_name, args.num_workers)
    agent.train()
