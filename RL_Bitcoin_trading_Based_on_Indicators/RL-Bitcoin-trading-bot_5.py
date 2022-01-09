# ================================================================
#
#   File name   : RL-Bitcoin-trading-bot_5.py
#   Author      : PyLessons
#   Created date: 2021-01-20
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : Trading Crypto with Reinforcement Learning #5
#
#   Code revised by: Alireza Alikhani, Sana Rastgar
#   Email       : alireza.alikhani@outlook.com 
#   Version     : 1.0.1
#
#
# ================================================================
from indicators import AddIndicators
from datetime import datetime
import matplotlib.pyplot as plt
from utils import TradingGraph
from model import Base_CNN, Critic_Model
from tensorflow.keras.optimizers import Adam
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

#import yfinance as yf
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

plt.figure(figsize=(10, 10))

class CustomAgent:
    # A custom Bitcoin trading agent
    def __init__(self, lookback_window_size=50, learning_rate=0.00005, epochs=1, optimizer=Adam, batch_size=32, models=[],state_size=10):
        self.lookback_window_size = lookback_window_size
        self.models = models


        #TODO: -1,0,1 looks more clean
        # Action space from 0 to 3, 0 is hold, 1 is buy, 2 is sell
        self.action_space = np.array([0, 1, 2])

        self.Critic = Critic_Model(self.state_size,self.action_space, learning_rate, optimizer)

        # folder to save models
        self.log_name = datetime.now().strftime("%Y_%m_%d_%H_%M")+"_Crypto_trader"

        self.state_size = (lookback_window_size, state_size)

        # Neural Networks part bellow
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimizer = optimizer
        self.batch_size = batch_size

        # TODO: add all base models
        ## Create Base network models
        self.Base = Base_CNN(
            input_shape=self.state_size, action_space=self.action_space.shape[0], learning_rate = self.learning_rate, optimizer=self.optimizer, models=self.models)
        
        ## Variables to keep the folder name and file name
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
        with open(self.log_name+"/Conditions.txt", "w") as conditions:
          indicators = inspect.getsource(CustomEnv.reset)
          indicators = indicators.split("# ____This is a Seperator not a comment____")
          conditions.write(indicators[1])

          main_code = inspect.getsource(main)
          main_code = main_code.split("# ____This is a Seperator not a comment____")
          conditions.write(main_code[1])

          train_agent_code = inspect.getsource(train_agent)
          train_agent_code = train_agent_code.split("# ____This is a Seperator not a comment____")
          conditions.write(train_agent_code[1])

          punishment = inspect.getsource(CustomEnv.get_reward)
          punishment = punishment.split("# ____This is a Seperator not a comment____")
          conditions.write(punishment[1])

        #   Actor = inspect.getsource(Actor_Model)
        #   Actor = Actor.split("# ____This is a Seperator not a comment____")
        #   conditions.write(Actor[1])
          
        #   Critic = inspect.getsource(Critic_Model)
        #   Critic = Critic.split("# ____This is a Seperator not a comment____")
        #   conditions.write(Critic[1])

    # TODO: Th is this?
    def get_gaes(self, rewards, dones, values, next_values, gamma=0.99, lamda=0.95, normalize=True):
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
        ## reshape memory to appropriate shape for training
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        predictions = np.vstack(predictions)

        ## Get Critic network predictions
        values = self.Critic.critic_predict(states)
        next_values = self.Critic.critic_predict(next_states)

        ## Compute advantages
        advantages, target = self.get_gaes(
            rewards, dones, np.squeeze(values), np.squeeze(next_values))

        ## stack everything to numpy array
        y_true = np.hstack([advantages, predictions, actions])

        # TODO: fit all base models (?)
        ## training Actor and Critic networks
        History = self.Base.fit(
            states, y_true, epochs=self.epochs, verbose=0, shuffle=True, batch_size=self.batch_size)

        return History.history['loss']

    def act(self, states):
        # Use the network to predict the next action to take, using the model

        # TODO: gather all predictions (?)
        prediction = self.Base.predict(np.expand_dims(states, axis=0))[0]
        action = np.random.choice(self.action_space, p=prediction)
        return action, prediction

    def save(self, name="Crypto_trader", score="", args=[]):
        # save keras model weights
        # TODO: Save all base models and overall model (?)
        self.file_name = f"{self.log_name}/{score}_{name}"
        self.Base.save_weights(
            f"{self.log_name}/{score}_{name}.h5")

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
        self.Base.load_weights(os.path.join(folder, f"{name}.h5"))


class CustomEnv:
    # A custom Bitcoin trading environment
    def __init__(self, df, initial_balance=1000, lookback_window_size=50, Render_range=100, Show_reward=False, Show_indicators=False, normalize_value=40000, indicator_index='MACD_1'):
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
        self.indicator_index = indicator_index
        self.indicator_indexes = {
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
        self.visualization = TradingGraph(Render_range=180, Show_reward=self.Show_reward,
                                          Show_indicators=self.Show_indicators)  # init visualization
        # limited orders memory for visualization
        self.trades = deque(maxlen=180)

        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.crypto_held = 0
        self.crypto_sold = 0
        self.crypto_bought = 0
        self.episode_orders = 0  # track episode orders count
        self.prev_episode_orders = 0  # track previous episode orders count
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
                [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])
            
            # TODO: these can be deleted
            self.market_history.append([self.df.loc[current_step, 'Open'],
                                        self.df.loc[current_step, 'High'],
                                        self.df.loc[current_step, 'Low'],
                                        self.df.loc[current_step, 'Close'],
                                        self.df.loc[current_step, 'Volume'],
                                        ])


            self.indicators_history.append(
                [
                    self.df.loc[current_step, self.indicator_index]/self.indicator_indexes[self.indicator_index],
                 ])

        # TODO: change these accordingly
        state = np.concatenate(
            (self.market_history, self.orders_history), axis=1) / self.normalize_value
        state = np.concatenate((state, self.indicators_history), axis=1)

        return state

    # TODO: these can be deleted
    # Get the data points for the given current_step
    def _next_observation(self):
        self.market_history.append([self.df.loc[self.current_step, 'Open'],
                                    self.df.loc[self.current_step, 'High'],
                                    self.df.loc[self.current_step, 'Low'],
                                    self.df.loc[self.current_step, 'Close'],
                                    self.df.loc[self.current_step, 'Volume'],
                                    ])

        self.indicators_history.append( 
                [
                    self.df.loc[self.current_step, self.indicator_index]/self.indicator_indexes[self.indicator_index]
                 ])

        obs = np.concatenate(
            (self.market_history, self.orders_history), axis=1) / self.normalize_value
        obs = np.concatenate((obs, self.indicators_history), axis=1)

        return obs

    ## Execute one time step within the environment
    def step(self, action):
        self.crypto_bought = 0
        self.crypto_sold = 0
        self.current_step += 1

        # TODO: these can be deleted
        ## Set the current price to Open
        current_price = self.df.loc[self.current_step, 'Open']
        Date = self.df.loc[self.current_step, 'Date']  # for visualization
        High = self.df.loc[self.current_step, 'High']  # for visualization
        Low = self.df.loc[self.current_step, 'Low']  # for visualization

        # TODO: -1,0,1 if it's not classification/ or even string maybe (?)
        if action == 0:  # Hold
            pass

        elif action == 1 and self.balance > self.initial_balance/100:
            # Buy with 100% of current balance
            self.crypto_bought = self.balance / current_price
            self.balance -= self.crypto_bought * current_price
            self.crypto_held += self.crypto_bought
            self.trades.append({'Date': Date, 'High': High, 'Low': Low,
                               'total': self.crypto_bought, 'type': "buy", 'current_price': current_price})
            self.episode_orders += 1

        elif action == 2 and self.crypto_held > 0:
            # Sell 100% of current crypto held
            self.crypto_sold = self.crypto_held
            self.balance += self.crypto_sold * current_price
            self.crypto_held -= self.crypto_sold
            self.trades.append({'Date': Date, 'High': High, 'Low': Low,
                               'total': self.crypto_sold, 'type': "sell", 'current_price': current_price})
            self.episode_orders += 1

        self.prev_net_worth = self.net_worth
        self.net_worth = self.balance + self.crypto_held * current_price

        self.orders_history.append(
            [self.balance, self.net_worth, self.crypto_bought, self.crypto_sold, self.crypto_held])

        # Receive calculated reward
        reward = self.get_reward()

        if self.net_worth <= self.initial_balance/2:
            done = True
        else:
            done = False

        obs = self._next_observation()

        return obs, reward, done

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
                    #self.trades[-2]['total']*self.trades[-1]['current_price']
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
        else:
            return 0 - self.punish_value

    # render environment
    def render(self, visualize=False):
        if visualize:
            ## Render the environment to the screen (inside utils.py file)
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
        states 
        for env in envs:
            state = env.reset(env_steps_size=training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            # env.render(visualize)
            action, prediction = agent.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state

        History = agent.replay(
            states, actions, rewards, predictions, dones, next_states)

        total_average.append(env.net_worth)
        average = np.average(total_average)

        agent.writer.add_scalar('Data/average net_worth', average, episode)
        agent.writer.add_scalar('Data/episode_orders',
                                env.episode_orders, episode)

        print("episode: {:<5} net worth {:<7.2f} average: {:<7.2f} orders: {}".format(
            episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                agent.save(score="{:.2f}".format(best_average), args=[
                           episode, average, env.episode_orders, a_loss, c_loss])
            agent.save()

    agent.end_training_log()


def test_agent(env, agent, visualize=True, test_episodes=10, folder="", name="Crypto_trader", comment=""):
    agent.load(folder, name)
    average_net_worth = 0
    average_orders = 0
    no_profit_episodes = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize and (episode == (test_episodes-1)))
            action, prediction = agent.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                average_orders += env.episode_orders
                if env.net_worth < env.initial_balance:
                    # calculate episode count where we had negative profit through episode
                    no_profit_episodes += 1
                print("episode: {:<5}, net_worth: {:<7.2f}, average_net_worth: {:<7.2f}, orders: {}".format(
                    episode, env.net_worth, average_net_worth/(episode+1), env.episode_orders))
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

    ## Reading a time-based dataframe with/without indicators
    df = pd.read_csv('./Binance_BTCUSDT_1h_Base_MACD_PSAR_ATR_BB_ADX_RSI_ICHI_Cnst_Interpolated.csv')  # [::-1]
    #df = df.sort_values('Date').reset_index(drop=True)
    #df = AddIndicators(df)  # insert indicators to df
    #df = df.round(2)   # two digit precision


    lookback_window_size = 12
    test_window = 24 * 30    # 30 days

    ## Training Section:
    train_df = df[:-test_window-lookback_window_size]
    agent = CustomAgent(lookback_window_size=lookback_window_size,
                        learning_rate=0.0001, epochs=5, optimizer=Adam, batch_size=24
                                                        , models=["CNN_1","CNN_2"], state_size=10+4)

    train_env1 = CustomEnv(train_df, lookback_window_size=lookback_window_size, indicator_index='MACD_1')
    train_env2 = CustomEnv(train_df, lookback_window_size=lookback_window_size, indicator_index='MACD_2')

    train_envs = [train_env1, train_env2]
    train_agent(train_envs, agent, visualize=False,
              train_episodes=2000, training_batch_size=500)
    
    agent.write_conditions()
    
    ## Testing Section:
    test_df = df[-test_window:-test_window + 180]
    ic(test_df[['Open','Close']])   # Depicting the specified Time-period
    test_env = CustomEnv(test_df, lookback_window_size=lookback_window_size,
                         Show_reward=True, Show_indicators=True)
    #test_agent(test_env, agent, visualize=True, test_episodes=100,
    #            folder="2021_10_06_11_42_Crypto_trader", name="1498.99_Crypto_trader", comment="")

