from matplotlib import axis
from indicators import AddIndicators
from datetime import datetime
import matplotlib.pyplot as plt
from utils import TradingGraph
from model import Base_CNN, Critic_Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from collections import deque
import random
import numpy as np
import pandas as pd
import inspect
import copy
import os
from icecream import ic
import re
from numpy.core.numeric import NaN
from pathlib import Path

#import yfinance as yf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
BUY, SELL = 0, 1
PHI = 0.999
plt.figure(figsize=(10, 10))


class CustomAgent:
    def __init__(self, lookback_window_size=50, learning_rate=0.00005, epochs=1, optimizer=Adam, batch_size=32, models=[], state_size=[]):
        self.lookback_window_size = lookback_window_size
        self.models = models

        # TODO: -1,0,1 looks more clean
        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1])

        self.state_size = []
        for size in state_size:
            self.state_size.append((lookback_window_size, size))

        # first size is critic's input size
        self.Critic = Critic_Model(
            self.state_size[0], self.action_space, learning_rate, optimizer)

        # folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_Crypto_trader"

        # Neural Networks part bellow
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # Create Base network models
        self.Base = Base_CNN(
            input_shapes=self.state_size[1:], action_space=self.action_space.shape[0], metalearner_action_space=3, learning_rate=self.learning_rate, optimizer=self.optimizer, models=self.models)

        # Variables to keep the folder name and file name
        self.folder_name = ""
        self.file_name = ""

    def get_folder_name(self):
        return self.log_name

    def start_training_log(self, initial_balance, normalize_value, train_episodes):
        # save training parameters to Parameters.txt file for future
        with open(self.log_name+"/Parameters.txt", "w") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training start: {current_date}\n")
            params.write(f"initial_balance: {initial_balance}\n")
            params.write(f"training episodes: {train_episodes}\n")
            params.write(
                f"lookback_window_size: {self.lookback_window_size}\n")
            params.write(f"learning_rate: {self.learning_rate}\n")
            params.write(f"epochs: {self.epochs}\n")
            params.write(f"batch size: {self.batch_size}\n")
            params.write(f"normalize_value: {normalize_value}\n")
            params.write(f"model: {self.model}\n")

    def end_training_log(self):
        with open(self.log_name+"/Parameters.txt", "a+") as params:
            current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
            params.write(f"training end: {current_date}\n")

    def write_conditions(self):
        with open(self.log_name+"/Conditions.txt", "w+") as conditions:
            indicators = inspect.getsource(CustomEnv.reset)
            indicators = indicators.split(
                "# ____This is a Seperator not a comment____")
            conditions.write(indicators[1])

            main_code = inspect.getsource(main)
            main_code = main_code.split(
                "# ____This is a Seperator not a comment____")
            conditions.write(main_code[1])

            train_agent_code = inspect.getsource(train_agent)
            train_agent_code = train_agent_code.split(
                "# ____This is a Seperator not a comment____")
            conditions.write(train_agent_code[1])

            punishment = inspect.getsource(CustomEnv.get_reward)
            punishment = punishment.split(
                "# ____This is a Seperator not a comment____")
            conditions.write(punishment[1])

        #   Actor = inspect.getsource(Actor_Model)
        #   Actor = Actor.split("# ____This is a Seperator not a comment____")
        #   conditions.write(Actor[1])

        #   Critic = inspect.getsource(Critic_Model)
        #   Critic = Critic.split("# ____This is a Seperator not a comment____")
        #   conditions.write(Critic[1])

    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):

        # TODO: why 2 fors and a copy? couldn't it be just one?

        deltas = [r + gamma * (1 - d) * nv - v for r, d,
                  nv, v in zip(rewards, dones, next_values, values)]

        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, predictions, dones, next_states):
        # reshape memory to appropriate shape for training
        states = np.vstack(states)
        # states = np.squeeze(states)

        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        # Get Critic network predictions
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)

        # Compute advantages
        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values))

        # stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])

        # TODO: fit all base models (?)
        # training Actor and Critic networks
        critic_history = self.Critic.Critic.fit(
            states, target, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        # History = self.Base.Base_Model.fit(
        #     [states for _ in range(len(self.models))], y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        History = self.Base.fit(
            [states for _ in range(len(self.models))], y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        return History.history['loss']

    def act(self, states):
        # Use the network to predict the next action to take, using the model

        # TODO: gather all predictions (?)
        # prediction = self.Base.predict(np.expand_dims(states, axis=1))[0]
        prediction = self.Base.predict(np.array(states))[0]

        buy = prediction[1] + (0.5 * prediction[0])
        adjusted_prediction = [buy, 1-buy]
        action = np.random.choice(self.action_space, p=adjusted_prediction)
        return action, prediction

    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        # TODO: Save all base models and overall model (?)
        self.file_name = f"{self.log_name}/{score}_{name}"
        Path(self.log_name).mkdir(parents=True, exist_ok=True)

        for i in range(len(self.Base.models)):
            self.Base.models[i]['model'].save(f"{self.file_name}_{i}")
        self.Base.Base_Model.save(f"{self.file_name}_metalearner")

        # self.Base.Base_Model.save_weights(
        #     f"{self.log_name}/{score}_{name}.h5")

        # TODO: Does this matter?
        # log saved model arguments to file
        if len(args) > 0:
            with open(f"{self.log_name}/log.txt", "a+") as log:
                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                atgumets = ""
                for arg in args:
                    atgumets += f", {arg}"
                log.write(f"{current_time}{atgumets}\n")

    def get_file_name(self):
        return self.file_name

    def load(self, folder, name):
        # load keras model weights
        # self.Base.Base_Model.load_weights(os.path.join(folder, f"{name}.h5"))
        files = os.listdir(folder)
        models = []
        for file in files:
            if name in file:
                models.append(file)
        models.sort()
        for i, model in enumerate(models[:-1]):
            self.Base.models[i]['model'] = load_model(
                f"{folder}/{model}", custom_objects={"ppo_loss": self.Base.ppo_loss})
        self.Base.Base_Model = load_model(
            f"{folder}/{models[-1]}", custom_objects={"ppo_loss": self.Base.ppo_loss})


class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, Show_indicators=False, normalize_value=40000, indicators=[], OHCL=0):
        # Define action space and state size and other custom parameters
        self.df = df.dropna().reset_index()
        self.df_total_steps = len(self.df) - 1
        self.initial_balance = initial_balance
        self.lookback_window_size = lookback_window_size
        self.Render_range = Render_range  # render range in visualization
        self.Show_reward = Show_reward  # show order reward in rendered visualization
        # show main indicators in rendered visualization
        self.Show_indicators = Show_indicators

        # Orders history contains the balance, net_worth, crypto_bought, crypto_sold, crypto_held values for the last lookback_window_size steps
        self.orders_history = deque(maxlen=self.lookback_window_size)

        # TODO: these can be deleted
        # Market history contains the OHCL values for the last lookback_window_size prices
        self.market_history = deque(maxlen=self.lookback_window_size)

        self.indicators_history = deque(maxlen=self.lookback_window_size)

        self.normalize_value = normalize_value
        self.OHCL = OHCL
        self.indicators = indicators
        self.indicator_indexes = {
            'macd_4h': 400,
            'macd_8h': 400,
            'williams_4h': 40000,
            'williams_8h': 40000,
            'MACD_1': 400,
            'MACD_4': 100,
            'MACD_2': 100,
            'psar_1': 1000,
            'psar_2': 40000,
            'psar_4': 40000,
            'psar_8': 40000,
            'ATR_2': 100,
            'ATR_4': 100,
            'bb_bbh_1': 40000,
            'bb_bbl_1': 40000,
            'bb_bbm_1': 40000,
            'bb_bbh_2': 40000,
            'bb_bbl_2': 40000,
            'bb_bbm_2': 40000,
            'bb_bbh_4': 40000,
            'bb_bbl_4': 40000,
            'bb_bbm_4': 40000,
            'ADX_1': 40,
            'RSI_1': 80,
            'ichi_a_1': 40000,
            'ichi_b_1': 40000,
            'ichi_base_line_1': 40000,
            'ichi_conversion_line_1': 40000,

        }

    # Reset the state of the environment to an initial state
    def reset(self, env_steps_size=0):
        self.trades = deque(maxlen=self.Render_range)
        self.action_history = deque(maxlen=2)
        self.action_history.append(0)
        self.action_history.append(0)
        self.total_hold = 0
        self.balance_long = self.initial_balance
        self.balance_short = self.initial_balance
        self.net_worth_long, self.net_worth_short = self.initial_balance, self.initial_balance
        self.net_worth = self.initial_balance * 2
        self.prev_net_worth = self.initial_balance
        self.crypto_enter_long = 0
        self.crypto_enter_short = 0
        self.crypto_transaction_long = 0
        self.crypto_held_long = 0
        self.crypto_sold = 0
        self.price_on_buy = 0
        self.price_on_sell = 0
        self.previous_price = 0
        self.last_trade_action = 0
        self.crypto_bought_long = 0
        self.setps_on_action = 0
        self.training_batch_size = 0
        self.batch_step = 0
        self.episode_orders = 0  # track episode orders count
        self.prev_episode_orders = 0  # track previous episode orders count
        self.previous_action = -1
        self.rewards = deque(maxlen=self.Render_range)
        self.env_steps_size = env_steps_size
        self.punish_value = 0
        if env_steps_size > 0:  # used for training dataset
            self.start_step = random.randint(
                self.lookback_window_size, self.df_total_steps - env_steps_size)
            self.end_step = self.start_step + env_steps_size
        else:  # used for testing dataset
            self.start_step = self.lookback_window_size
            self.end_step = self.df_total_steps

        self.current_step = self.start_step

        for i in reversed(range(self.lookback_window_size)):
            current_step = self.current_step - i
            self.orders_history.append(
                # [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
                [self.balance_long, self.balance_short, self.net_worth_long, self.net_worth_short,
                 self.net_worth, self.crypto_enter_long, self.crypto_enter_short])
            temp = []
            for indicator in self.indicators:
                temp.append(
                    self.df.loc[current_step, indicator] /
                    self.indicator_indexes[indicator],
                )

            self.indicators_history.append(temp)

            # TODO: these can be deleted
            if(self.OHCL == 1):
                self.market_history.append([self.df.loc[current_step, 'open'],
                                            self.df.loc[current_step, 'high'],
                                            self.df.loc[current_step, 'low'],
                                            self.df.loc[current_step, 'close'],
                                            self.df.loc[current_step,
                                                        'volume'],
                                            ])
                state = np.concatenate(
                    (self.market_history, self.orders_history), axis=1) / self.normalize_value
                state = np.concatenate(
                    (state, self.indicators_history), axis=1)
            else:
                state = self.indicators_history

        # TODO: change these accordingly
        # print(self.market_history.shape, self.indicators_history.shape)

        return state

    # TODO: these can be deleted
    # Get the data points for the given current_step
    def _next_observation(self):
        temp = []
        for indicator in self.indicators:
            temp.append(
                self.df.loc[self.current_step, indicator] /
                self.indicator_indexes[indicator],
            )

        self.indicators_history.append(temp)

        if(self.OHCL == 1):
            self.market_history.append([self.df.loc[self.current_step, 'open'],
                                        self.df.loc[self.current_step, 'high'],
                                        self.df.loc[self.current_step, 'low'],
                                        self.df.loc[self.current_step,
                                                    'close'],
                                        self.df.loc[self.current_step,
                                                    'volume'],
                                        ])
            obs = np.concatenate(
                (self.market_history, self.orders_history), axis=1) / self.normalize_value
            obs = np.concatenate((obs, self.indicators_history), axis=1)

        else:
            obs = self.indicators_history

        return obs

    # Execute one time step within the environment
    def step(self, action, shift_strategy):
        self.current_step += 1
        self.batch_step += 1

        # Set the current price to Open
        current_price = self.df.loc[self.current_step, 'open']
        date = self.df.loc[self.current_step, 'date']  # for visualization
        High = self.df.loc[self.current_step, 'high']  # for visualization
        Low = self.df.loc[self.current_step, 'low']  # for visualization

        if self.previous_action != action:
            # ic('enter action')

            if action == 0:
                # ready_to_reward = True
                # ic('action')
                self.price_on_buy = current_price
                self.last_trade_action = BUY

                # Long Section
                self.crypto_enter_long = (
                    self.balance_long / current_price)*PHI
                self.balance_long = 0
                self.trades.append({'date': date, 'high': High, 'low': Low,
                                    'total_crypto': self.crypto_enter_long, 'type': "long", 'current_price': current_price})

                # Short Section
                if self.previous_action != -1:
                    self.balance_short += self.crypto_enter_short * \
                        (self.price_on_sell*2-current_price)
                    self.balance_short *= PHI
                    self.crypto_enter_short = 0

            elif action == 1:
                # ready_to_reward = True
                self.last_trade_action = SELL
                self.price_on_sell = current_price

                # Long Section
                if self.previous_action != -1:
                    self.balance_long += self.crypto_enter_long * current_price
                    self.balance_long *= PHI
                    self.crypto_enter_long = 0

                # Short Section
                self.crypto_enter_short = self.balance_short / current_price
                self.crypto_enter_short *= PHI
                self.balance_short = 0
                self.trades.append({'date': date, 'high': High, 'low': Low,
                                    'total_crypto': self.crypto_enter_short, 'type': "short", 'current_price': current_price})

            self.episode_orders += 1
            self.previous_action = action

        else:
            pass
            # ready_to_reward = False

        self.net_worth_long = self.balance_long + \
            self.crypto_enter_long * current_price
        self.net_worth_short = self.balance_short + \
            self.crypto_enter_short * (self.price_on_sell*2-current_price)

        self.prev_net_worth = self.net_worth
        # ic(self.net_worth_long)
        # ic(self.net_worth_short)

        self.net_worth = self.net_worth_long + self.net_worth_short
        # ic(self.net_worth)

        self.orders_history.append(
            [self.balance_long, self.balance_short, self.net_worth_long, self.net_worth_short,
             self.net_worth, self.crypto_enter_long, self.crypto_enter_short])

        # Receive calculated reward
        if shift_strategy:
            self.setps_on_action += 1
            # reward, ready_to_reward = self.get_reward_strategy_2(current_price)
            reward = self.get_reward()
        else:
            # reward = self.get_reward_strategy_1(current_price)
            reward = self.get_reward()
        self.prev_episode_orders = self.episode_orders
        self.action_history.append(action)
        self.previous_price = current_price

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done

    def get_reward_strategy_1(self, current_price):
        self.punish_value += self.net_worth * 0.00005
        # ic(action)
        # ic(current_price)
        # ic(self.previous_price)
        reward = 0

        if self.episode_orders > 1:
            # self.action_history[0] changed to action
            if self.action_history[0] == BUY:
                # ic('buy')

                reward = self.net_worth_long * \
                    np.log(current_price/self.previous_price)
                if self.action_history[0] == self.action_history[1]:
                    reward -= self.punish_value
                else:
                    self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                # ic(reward)
                return reward
            elif self.action_history[0] == SELL:
                # ic('sell')
                reward = self.net_worth_short * \
                    np.log(self.previous_price/current_price)
                if self.action_history[0] == self.action_history[1]:
                    reward -= self.punish_value
                else:
                    self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                # ic(reward)
                return reward

        return 0 - self.punish_value

    def get_reward_strategy_2(self, current_price):
        self.punish_value += self.net_worth * 0.00005
        reward = 0

        if self.batch_step > 1:
            if self.previous_action != self.action_history[1]:
                if self.action_history[1] == BUY:

                    reward = self.net_worth_long * \
                        np.log(current_price/self.price_on_buy)
                    reward /= self.setps_on_action
                    self.punish_value = 0
                    self.trades[-1]["Reward"] = reward
                    rewards = [reward]*self.setps_on_action
                    self.setps_on_action = 0
                    ready_to_reward = True
                    # ic(reward)
                    return (rewards, ready_to_reward)
                elif self.action_history[1] == SELL:
                    # ic('sell')
                    reward = self.net_worth_short * \
                        np.log(self.price_on_sell/current_price)
                    reward /= self.setps_on_action
                    self.punish_value = 0
                    self.trades[-1]["Reward"] = reward
                    # ic(reward)
                    rewards = [reward]*self.setps_on_action
                    self.setps_on_action = 0
                    ready_to_reward = True
                    return (rewards, ready_to_reward)

            elif self.batch_step == self.training_batch_size:
                if self.last_trade_action == BUY:
                    reward = self.net_worth_short * \
                        np.log(self.price_on_buy/current_price)
                    reward /= self.setps_on_action
                    rewards = [reward]*self.setps_on_action

                else:
                    reward = self.net_worth_long * \
                        np.log(current_price/self.price_on_sell)
                    reward /= self.setps_on_action
                    rewards = [reward]*self.setps_on_action

                self.setps_on_action = 0
                ready_to_reward = True
                return (rewards, ready_to_reward)
            else:
                return (None, False)
        else:
            return (None, False)

    # Calculate reward

    def get_reward(self):
        self.punish_value += self.net_worth * 0.000002
        #self.punish_value += self.net_worth * self.punishment_cofactor
        #self.punish_value = 0
        if self.episode_orders > 1 and self.episode_orders > self.prev_episode_orders:
            # <--Just covers Sell-Buy and Buy-Sell, not others -->
            self.prev_episode_orders = self.episode_orders
            if self.trades[-1]['type'] == "buy" and self.trades[-2]['type'] == "sell":
                reward = self.trades[-2]['total']*self.trades[-2]['current_price'] - \
                    self.trades[-1]['total']*self.trades[-1]['current_price']
                # self.trades[-2]['total']*self.trades[-1]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
            elif self.trades[-1]['type'] == "sell" and self.trades[-2]['type'] == "buy":
                reward = self.trades[-1]['total']*self.trades[-1]['current_price'] - \
                    self.trades[-2]['total']*self.trades[-2]['current_price']
                reward -= self.punish_value
                self.punish_value = 0
                self.trades[-1]["Reward"] = reward
                return reward
            return 0 - self.punish_value
        else:
            return 0 - self.punish_value

    # render environment

    def render(self, visualize=False):
        if visualize:
            # Render the environment to the screen (inside utils.py file)
            img = self.visualization.render(
                self.df.loc[self.current_step], self.net_worth, self.trades)
            return img


def train_agent(envs, agent, visualize=False, train_episodes=50, training_batch_size=500):
    initial_balance = 1000
    normalize_value = 40000
    # agent.create_writer(initial_balance, normalize_value,
    #                     train_episodes)  # create TensorBoard writer

    total_average = deque(maxlen=2)
    best_average = 0  # used to track best average net worth
    for episode in range(train_episodes):
        reset_states = []
        for env in envs:
            reset_states.append(env.reset(env_steps_size=training_batch_size))

        states = [[] for _ in range(len(envs))]
        actions, rewards, predictions, dones, next_states = [], [], [], [], []
        for t in range(training_batch_size):
            # env.render(visualize)
            action, prediction = agent.act(reset_states)
            for i, env in enumerate(envs):
                next_state, reward, done = env.step(action, True)
                # states[i].append(np.expand_dims(reset_states[i], axis=0))
                states[i].append(np.array(reset_states[i]))
                next_states.append(np.expand_dims(next_state, axis=0))
                reset_states[i] = next_state
                rewards.append(reward)
                dones.append(done)
                action_onehot = np.zeros(len(agent.action_space))
                action_onehot[action] = 1
                actions.append(action_onehot)
                predictions.append(prediction)

        history = agent.replay(
            states, actions, rewards, predictions, dones, next_states)

        temp = 0
        for env in envs:
            temp += env.net_worth

        total_average.append(temp)
        average = np.average(total_average)

        # agent.writer.add_scalar('Data/average net_worth', average, episode)
        # agent.writer.add_scalar('Data/episode_orders',
        #                         env.episode_orders, episode)

        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(
            episode, total_average[-1], average, envs[0].episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("saving model")
                agent.save(score="{:.2f}".format(best_average), args=[
                           episode, average, envs[0].episode_orders, history])
            agent.save()

    agent.end_training_log()


def test_agent(envs, agent, visualize=True, test_episodes=10, folder="", name="Crypto_trader", comment=""):
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        states = []
        for env in envs:
            states.append(env.reset())
        while True:
            loop_var = False
            for env in envs:
                env.render(visualize and (episode == (test_episodes-1)))
                action, prediction = agent.act(states)
                state, reward, done = env.step(action, True)
                if env.current_step == env.end_step:
                    loop_var = True
                    average_net_worth += env.net_worth
                    average_orders += env.episode_orders
                    # calculate episode count where we had negative profit through episode
                    if env.net_worth < env.initial_balance:
                        no_profit_episodes += 1
                    print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(
                        episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
                    break
            if loop_var:
                break

    print("average {} episodes agent net_worth: {}, orders: {}".format(
        test_episodes, average_net_worth/test_episodes, average_orders/test_episodes))
    print("No profit episodes: {}".format(no_profit_episodes))
    # save test results to test_results.txt file
    with open("test_results.txt", "a+") as results:
        current_date = datetime.now().strftime('%Y-%m-%d %H:%M')
        results.write(f'{current_date}, {name}, test episodes:{test_episodes}')
        results.write(
            f', net worth:{average_net_worth/(episode+1)}, orders per episode:{average_orders/test_episodes}')
        results.write(
            f', no profit episodes:{no_profit_episodes}, model: {agent.model}, comment: {comment}\n')


if __name__ == "__main__":

    # Reading a time-based dataframe with/without indicators
    df = pd.read_csv('./Binance_BTCUSDT_ALL_INDCTRS_4H_2019.csv')  # [::-1]
    df2 = pd.read_csv('./Binance_BTCUSDT_PSAR_MACD_8H_2019.csv')  # [::-1]
    #df = df.sort_values('Date').reset_index(drop=True)
    # df = AddIndicators(df)  # insert indicators to df
    # df = df.round(2)   # two digit precision

    lookback_window_size = 12
    test_window = 24 * 30    # 30 days

    # Training Section:
    train_df = df[:-test_window-lookback_window_size]
    train_df2 = df2[:-test_window-lookback_window_size]
    agent = CustomAgent(lookback_window_size=lookback_window_size,
                        learning_rate=0.0001, epochs=5, optimizer=Adam, batch_size=30, models=['B1000/model4h_1_actor.h5', 'B1000/model8h_1_actor.h5'], state_size=[14, 14, 14])

    #train_env1 = CustomEnv(train_df, lookback_window_size=lookback_window_size, indicator_index='MACD_1')
    #train_env2 = CustomEnv(train_df, lookback_window_size=lookback_window_size, indicator_index='MACD_2')
    train_env = CustomEnv(train_df, lookback_window_size=lookback_window_size, indicators=[
                          'macd_4h', 'williams_4h'], OHCL=1)
    train_env2 = CustomEnv(train_df2, lookback_window_size=lookback_window_size, indicators=[
                           'macd_8h', 'williams_8h'], OHCL=1)

    #train_envs = [train_env1, train_env2]

    train_agent([train_env, train_env2], agent, visualize=False,
                train_episodes=2000, training_batch_size=1000)

    # agent.write_conditions()

    # Testing Section:
    test_df = df[:-test_window]
    ic(test_df[['Open', 'Close']])   # Depicting the specified Time-period
    # test_env1 = CustomEnv(test_df, lookback_window_size=lookback_window_size, indicator_index='MACD_1')
    # test_env2 = CustomEnv(test_df, lookback_window_size=lookback_window_size, indicator_index='MACD_2')
    # test_agent([test_env1, test_env2], agent, visualize=True, test_episodes=100,
    #            folder="2022_02_08_19_00_Crypto_trader", name="2272.15_Crypto_trader", comment="")
