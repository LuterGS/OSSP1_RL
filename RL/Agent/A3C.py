from threading import Thread, Lock

import gym
import tensorflow as tf
import RL.Agent.ActorCriticModels as models
import numpy as np

CUR_EP = 0

class Actor:
    def __init__(self, state_dimension, action_dimension, action_bound, std_bound, learning_rate=0.0005, model_type="default"):

        self.state_dimension = state_dimension
        self.action_dimension = action_dimension
        self.action_bound = action_bound
        self.std_bound = std_bound

        if model_type == "default":
            self.model = models.Actor_Continuos_Default(state_dimension, action_dimension, action_bound)
        elif model_type == "IMG":
            self.model = models.Actor_Continuos_IMG(state_dimension, action_dimension, action_bound)
        else:
            self.model = models.Actor_Continuos_Default(state_dimension, action_dimension, action_bound)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.entropy_beta = 0.01


    def action(self, state):
        state = np.reshape(state, [1, self.state_dimension])    # 차원을 한 번 올리기만 함
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        # 에이전트의 결과값에서 실제 행동을 선택 (random)
        return np.random.normal(mu, std, size=self.action_dimension)

    def log_policy(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        # 입력으로 들어온 std를 std_bound에 맞춰줌 (더 낮거나 높은 값은 bound의 최소/최대로 맞춰줌)
        var = std ** 2  # 제곱
        log_policy_pdf = -0.5 * (action - mu) ** 2 / var - 0.5 * tf.math.log(var * 2 * np.pi)
            # 어...? 이거 policy network 관련해서 나왔던거같은데.
            # 수학적 공식이니 나중에 알아보자
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)
            # log_policy_pdf 의 각 축마다 합을 더함 (한 차원 축소)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_policy(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)
        # loss 계산인데, 음수로 진행하는걸 봐서, gradient descent의 한 방법인 듯

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        gradient_res = tape.gradient(loss, self.model.trainable_variables)
        # y = f(x), x = model.trainable_variables, y = loss, f'(x) 값을 의미함
        self.optimizer.apply_gradients(zip(gradient_res, self.model.trainable_variables))
        # 한 번의 학습을 거친 뒤 loss를 return
        return loss


class Critic:

    def __init__(self, state_dimension, learning_rate=0.001, model_type="default"):
        self.state_dimension = state_dimension

        if model_type == "IMG":
            self.model = models.Critic_CNN(state_dimension)
        else:
            self.model = models.Critic_Default(state_dimension)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.loss_computer = tf.keras.losses.MeanSquaredError()

    def compute_loss(self, v_prediction, td_targets):
        return self.loss_computer(td_targets, v_prediction)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_prediction = self.model(states, training=True)
            assert v_prediction.shape == td_targets.shape
            loss = self.compute_loss(v_prediction, tf.stop_gradient(td_targets))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        return loss


class Worker(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes, gamma, update_interval, worker_num):
        Thread.__init__(self)
        self.lock = Lock()
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim,self.action_bound, self.std_bound)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())

        self.gamma = gamma
        self.update_interval = update_interval
        self.worker_num = worker_num

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = self.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advantages(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, lists):
        batch = lists[0]
        for element in lists[1:]:
            batch = np.append(batch, element, axis=0)
        return batch

    def train(self):
        global CUR_EP
        while self.max_episodes >= CUR_EP:
            states, actions, rewards = [], [], []
            ep_reward, done = 0, False

            state = self.env.reset()

            while not done:
                action = self.actor.action(state)
                action = np.clip(action, -self.action_bound, self.action_bound)

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, 1])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if len(states) >= self.update_interval or done:
                    state_batch = self.list_to_batch(states)
                    action_batch = self.list_to_batch(actions)
                    reward_batch = self.list_to_batch(rewards)

                    next_v_val = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        (reward_batch + 8) / 8,  # 얘는 왜 이렇게 나오는지 한번 봐야할 듯
                        next_v_val, done)
                    advantages = td_targets - self.critic.model.predict(state_batch)

                    with self.lock:

                        self.global_actor.train(state_batch, action_batch, advantages)
                        self.global_critic.train(state_batch, td_targets)

                        self.actor.model.set_weights(self.global_actor.model.get_weights())
                        self.critic.model.set_weights(self.global_critic.model.get_weights())

                    states, actions, rewards = [], [], []

                ep_reward += reward[0][0]
                state = next_state[0]

            print("Cur Ep : ", CUR_EP, "\tCur Worker : ", self.worker_num, "\treward : ", ep_reward)
            CUR_EP += 1

    def run(self):
        self.train()





class A3C:
    def __init__(self, env: gym.Env, agent_workers: int, max_episode: int, gamma: float, update_interval: int):
        """
        :param env: 넣을 때 새로 생성해서 줘야 함!
        """
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_bound = env.action_space.high[0]
        self.std_bound = [1e-2, 1.0]

        self.global_actor = Actor(self.state_dim, self.action_dim, self.action_bound, self.std_bound)
        self.global_critic = Critic(self.state_dim)
        self.num_workers = agent_workers
        self.max_episode = max_episode
        self.gamma = gamma
        self.update_interval = update_interval

    def train(self, make_env):
        """

        :param make_env: 해당 함수를 실행하면 gym env를 return해줘야 함
        :return:
        """

        workers = []
        for i in range(self.num_workers):
            single_env = make_env()
            workers.append(Worker(single_env, self.global_actor, self.global_critic, self.max_episode, self.gamma, self.update_interval, i))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()



