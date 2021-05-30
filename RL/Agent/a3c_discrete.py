# import wandb
from datetime import datetime

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.layers import Input, Dense

import os
import gym
import argparse
import numpy as np
from threading import Thread, Lock
from multiprocessing import cpu_count

from RL.Environment.BasicGymEnv_process import BasicEnv

FILE_LOC = os.path.dirname(os.path.abspath(__file__))

tf.keras.backend.set_floatx('float64')


parser = argparse.ArgumentParser()
parser.add_argument("--env", default="TEST")
parser.add_argument("--num-workers", default=2, type=int)
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
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        # optimizer라고 보면 될 듯, args에서 받은 learning_rate를 설정함.

        # 아마 이게 엔트로피의 초기값을 설정해주는 것이라고 생각하면 될 듯
        self.entropy_beta = 0.01

    def nn_model(self):
        # 예정대로 모델을 만든다.
        # state를 토대로 action을 return하는 모델을 만듬
        return tf.keras.Sequential([
            layers.Input((480, 640,)),
            layers.Reshape((480, 640, 1, )),
            layers.Conv2D(32, 3, strides=(3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(4, activation='sigmoid')
        ])

    def compute_loss(self, actions, logits, advantages):
        ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        # ce_loss는 A2C에서도 봤던 기존의 loss를 의미한다.

        entropy_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # 이게 새로 추가된 부분인데, 엔트로피가 무엇인지 확인해봐야할 것 같다.

        actions = tf.cast(actions, tf.int32)
        # 마찬가지로, actions의 안의 자료형을 모두 int형으로 변환함 (이 부분을 추후 수정할 수도 있음)

        policy_loss = ce_loss(actions, logits, sample_weight=tf.stop_gradient(advantages))
        # actions가 에이전트가 실제로 행동한 정답이고, logits가 예측값, 즉 신경망의 예측값을 의미한다.
        # 이 두 개의 SparseCategoricalCrossentropy 를 계산한 값이 policy_loss 이다.
        # 결과에 advantages를 곱해주기만 한다 (advantages가 학습되는 것을 방지하기 위해 tf.stop_gradient를 사용

        entropy = entropy_loss(logits, logits)
        # 또한, entropy loss function에서 entropy loss를 구하기 위해,
        # logits와 logits를 넣는다? 이러면 loss가 아예 없지 않을까?
        # 아마 이건
        # entropy = entropy_loss(actions, logits) 로 수정해줘야할 것 같다.

        return policy_loss - self.entropy_beta * entropy
        # 기존 loss에서 entropy loss를 일정 비율만큼 제거한 값을 return해줌
        # (잘은 모르겠지만, 아마 한 에이전트의 영항이 과도해지는 것을 막기 위해서인 것 같다)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            logits = self.model(states, training=True)
            loss = self.compute_loss(
                actions, logits, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        # step function
        return loss

        # A2C에서 비교했던 loss function과 동일함.


class Critic:
    # 가치함수를 평가하는 Critic -> 판단가

    def __init__(self, state_dim):
        # A2C 와 동일
        self.state_dim = state_dim
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def nn_model(self):
        # A2C와 동일
        return tf.keras.Sequential([
            layers.Input(shape=(480, 640,)),
            layers.Reshape((480, 640, 1,)),
            layers.Conv2D(64, 10, strides=(10, 10,), activation='relu'),
            layers.Dropout(0.2),
            # layers.Conv2D(32, 3, activation='relu'),
            # layers.Dropout(0.2),
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        # A2C와 동일
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        # A2C와 동일
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class A3C:
    # Single Agent

    def __init__(self, env_func):
        self.env_func = env_func
        env = env_func()
        # A2C와는 다르게, 한 Agent가 한 Env를 가져야하기 때문에,
        # Gym의 이름만 받고, Agent 내부에서 Gym을 생성해 줌.

        self.state_dim = -1
        self.action_dim = 4

        self.global_actor = Actor(self.state_dim, self.action_dim)
        self.global_critic = Critic(self.state_dim)
        # 범용적인 액터 - 크리틱을 생성함.

        self.num_workers = args.num_workers
        # Actor의 개수는, cpu thread의 개수를 따름.

    def train(self, max_episodes=1000):
        # 실제로 A3C가 트레이닝되는데 있어서 어떻게 동작하는지.
        workers = []

        # CPU 스레드의 개수만큼 Agent들을 생성해주는데,
        # 각 에이전트마다 Gym 환경을 만들어준 후
        # 각 worker에 WorkerAgent (싱글 에이전트) 와, 글로벌 액터-크리틱, 최대 학습수를 리턴해준다.
        for i in range(self.num_workers):
            workers.append(
                WorkerAgent(self.env_func, self.global_actor, self.global_critic, max_episodes, i)
            )

        # 학습을 시작하고
        for worker in workers:
            worker.start()

        # 종료해준다.
        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes, i):
        Thread.__init__(self)
        # Lock을 Aquire 한다. 음...?
        self.lock = Lock()
        self.env = env()
        self.state_dim = -1
        self.action_dim = 4

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic

        self.i = i

        # 글로벌 신경망 말고, 개인 신경망도 만들어준다.
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim)

        # 또한, 초기값을 같게 하기 위해, 가중치를 글로벌 값과 동일하게 설정한다.
        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

    def n_step_td_target(self, rewards, next_v_value, done):
        # 아마 한 번의 step이 이뤄졌을 때 행동하는 것인듯?
        # 실제 reward에서 "할인된 보상"값을 더해서 return해주는 것이라고 생각하면 되겠다.

        td_targets = np.zeros_like(rewards)
        # reward 가 모두 0인 배열을 하나 만들어준다.

        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            # for k in range(len(rewards) - 1, -1, -1) 과 같음
            # 이렇게 굳이 하는 이유는... 지금까지 온 길을 백트래킹하면서 학습한다고 생각하면 될 듯.

            cumulative = args.gamma * cumulative + rewards[k]
            # 만약 에피소드가 완료되지 않으면, cumulative의 값은 gamma값에 next_v_value를 곱하고
            # 그 값에 진짜 reward를 더한 값

            # 에피소드가 완료되었으면, 그냥 reward와 동일하다.

            td_targets[k] = cumulative
            # 결과값에 저장 후 반환한다.
        return td_targets

    def advatnage(self, td_targets, baselines):
        # advantage 함수
        return td_targets - baselines

    def list_to_batch(self, list):
        # list를 numpy batch로 바꾸는 부분
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        global GLOBAL_EPISODE_NUM
        # 한 에이전트마다 max_episodes만큼 돈다고 생각하면 되는데...
        # 재밌는 점은, 저건 global variable이라서 변수들이 계속 스스로 공유할거란 말이지?
        # 그러면... 아마 워커수만큼 나눠진 학습을 할듯?

        while self.max_episodes >= GLOBAL_EPISODE_NUM:
            state_batch = []  # state 들
            action_batch = []  # batch 들
            reward_batch = []  # reward 들
            episode_reward, done = 0, False  # 초기 리워드 설정

            state = self.env.reset()  # 처음으로 env 초기화해줌

            while not done:
                # self.env.render()
                print(state, state.shape)
                state = np.reshape(state, [1, 480, 640])
                probs = self.actor.model.predict(state)
                print("probs : ", probs)
                action = probs[0]
                # state에 따른 신경망의 출력
                # 실제 에이전트의 행동을 구함

                next_state, reward, done, _ = self.env.step(action)
                # action을 기반으로 env에 action을 취함

                state = np.reshape(state, [1, 480, 640])
                action = np.reshape(action, [1, 4])
                next_state = np.reshape(next_state, [1, 480, 640])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    # 업데이트 주기가 오면...

                    states = np.array([state.squeeze() for state in state_batch])
                    actions = np.array([action.squeeze() for action in action_batch])
                    rewards = np.array([reward.squeeze() for reward in reward_batch])
                    # 값들을 배치로 전환해준 다음에

                    next_v_value = self.critic.model.predict(next_state)
                    # 다음 v값을 구하고

                    td_targets = self.n_step_td_target(
                        rewards, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)
                    # 그 v값을 기반으로 td_target값과, 가치함수값을 구한다.

                    with self.lock:
                        # 이후 글로벌 actor, critic을 업데이트하고, 그에 맞게 본인도 업데이트한다.

                        actor_loss = self.global_actor.train(
                            states, actions, advantages)
                        critic_loss = self.global_critic.train(
                            states, td_targets)

                        self.actor.model.set_weights(
                            self.global_actor.model.get_weights())
                        self.critic.model.set_weights(
                            self.global_critic.model.get_weights())

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []  # 얘네들은 안쓰네... 필요가 없는듯?
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print(f'Agent {self.i}, EP{GLOBAL_EPISODE_NUM}, EpisodeReward={episode_reward}')
            tf.summary.scalar("episode_reward", episode_reward, step=GLOBAL_EPISODE_NUM)
            GLOBAL_EPISODE_NUM += 1

    def run(self):
        self.train()


def main():
    agent = A3C(BasicEnv)
    agent.train()


if __name__ == "__main__":
    main()
