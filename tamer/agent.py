import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

from pynput import keyboard

MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}
# MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')

class SGDFunctionApproximatorDiscrete:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        # observation_examples = np.array(
        #     [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        # )
        observation_examples = np.array([[i] for i in range(env.observation_space.n)])
        print(observation_examples)
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            # print(self.featurize_state([env.reset()]))
            model.partial_fit([self.featurize_state([env.reset()])], [0])
            # print("env.reset()", env.reset())
            # model.partial_fit([[env.reset()]], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state([state])
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state([state])
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        # print(state)
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class SGDFunctionApproximatorContinuous:
    """ SGD function approximator with RBF preprocessing. """
    def __init__(self, env):
        
        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            model.partial_fit([self.featurize_state(env.reset())], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        # print(state)
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """
    def __init__(
        self,
        env,
        num_episodes,
        discount_factor=1,  # only affects Q-learning
        epsilon=0, # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        tame=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training TAMER
        output_dir=LOGS_DIR,
        model_file_to_load=None  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir

        # 1 phần thưởng trên 1 timestep
        self.rewarded_ts = False
        self.human_reward = 0

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = SGDFunctionApproximatorContinuous(env)  # init H function
                # self.H = SGDFunctionApproximatorDiscrete(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximatorContinuous(env)  # init Q function
                # self.Q = SGDFunctionApproximatorDiscrete(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # reward logging
        self.reward_log_columns = [
            'Episode',
            'Ep start ts',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]
        self.reward_log_path = os.path.join(self.output_dir, f'{self.uuid}.csv')

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def on_press(self, key):
        try:
            if not self.rewarded_ts:  # Chỉ xử lý khi chưa được thưởng
                if key.char == 'w': self.human_reward = 1
                if key.char == 'a': self.human_reward = -1
                self.rewarded_ts = True  # Đánh dấu đã được thưởng
        except AttributeError:
            pass

    def _train_episode(self, episode_index, disp):
        print(f'Episode: {episode_index + 1}')
        # print(f'  Timestep:')

        rng = np.random.default_rng()
        tot_reward = 0
        state = self.env.reset()
        ep_start_time = dt.datetime.now().time()

        listener = keyboard.Listener(on_press=self.on_press)
        listener.start()

        with open(self.reward_log_path, 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=self.reward_log_columns)
            dict_writer.writeheader()
            for ts in count():
                self.rewarded_ts = False
                self.human_reward = 0

                print(f'  ts: {ts} - ', end='')
                # self.env.render() # deprecated
                # print(f'state: {state} - ', end='')

                # Determine next action
                action = self.act(state)
                disp.show_action(action)
                # if self.tame:
                #     disp.show_action(action)

                # Get next state and reward
                next_state, reward, done, info = self.env.step(action)
                # print(f'next state: {next_state}')

                if not self.tame:
                    # if done and next_state[0] >= 0.5:
                    if done:
                        td_target = reward
                    else:
                        td_target = reward + self.discount_factor * np.max(
                            self.Q.predict(next_state)
                        )
                    self.Q.update(state, action, td_target)
                else:
                    now = time.time()
                    while time.time() < now + self.ts_len:
                        frame = None

                        time.sleep(0.01)  # save the CPU

                        # human_reward = disp.get_scalar_feedback()
                        # feedback_ts = dt.datetime.now().time()
                        if self.human_reward != 0:
                            print("    human reward:", self.human_reward)
                            dict_writer.writerow(
                                {
                                    'Episode': episode_index + 1,
                                    'Ep start ts': ep_start_time,
                                    # 'Feedback ts': feedback_ts,
                                    'Human Reward': self.human_reward,
                                    'Environment Reward': reward
                                }
                            )
                            self.H.update(state, action, self.human_reward)
                            self.human_reward = 0
                            # break

                tot_reward += reward
                if done:
                    print(f'  Reward: {tot_reward}')
                    break

                stdout.write('\b' * (len(str(ts)) + 1))
                state = next_state

        listener.stop()

        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step

    # async def train(self, model_file_to_save=None):
    def train(self, model_file_to_save=None):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        # render first so that pygame display shows up on top
        # self.env.render() # deprecated
        # self.env.reset()

        disp = None
        # if self.tame:
            # only init pygame display if we're actually training tamer
        from .interface import Interface
        disp = Interface(action_map=MOUNTAINCAR_ACTION_MAP)

        for i in range(self.num_episodes):
            self._train_episode(i, disp)

        print('\nCleaning up...')
        # self.env.close()
        # if model_file_to_save is not None:
        #     self.save_model(filename=model_file_to_save)

    def play(self, n_episodes=1):
        """
        Run episodes with trained agent
        Args:
            n_episodes: number of episodes
            render: optionally render episodes

        Returns: list of cumulative episode rewards
        """
        self.epsilon = 0
        ep_rewards = []
        for i in range(n_episodes):
            state = self.env.reset()
            done = False
            tot_reward = 0
            while not done:
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)
                tot_reward += reward
                state = next_state
            ep_rewards.append(tot_reward)
            print(f'Episode: {i + 1} Reward: {tot_reward}')
        # self.env.close()
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        self.env.close()
        return avg_reward

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model
