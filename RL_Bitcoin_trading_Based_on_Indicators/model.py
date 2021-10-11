#================================================================
#
#   File name   : model.py
#   Author      : PyLessons
#   Created date: 2021-01-13
#   Website     : https://pylessons.com/
#   GitHub      : https://github.com/pythonlessons/RL-Bitcoin-trading-bot
#   Description : defined PPO Keras model classes
#
#
#   Code revised by: Alireza Alikhani
#   Email       : alireza.alikhani@outlook.com 
#
#================================================================
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Dropout
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import dropout
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

class Shared_Model:
    def __init__(self, input_shape, action_space, learning_rate, optimizer, model="Dense"):
        X_input = Input(input_shape)
        self.action_space = action_space
            
        
        ## Critic model
        ### Dense Model (simple)
        '''V = Dense(512, activation="relu")(X)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)'''

        ### Dense Model + DropOut
        '''X = Flatten()(X_input)
        X = Dense(512, activation="relu")(X)
        #dropout_layer = Dropout(.2)
        
        #V = dropout_layer(X)
        V = Dense(512, activation="relu")(X)
        #V = dropout_layer(V)
        V = Dense(512, activation="relu")(V)
        #V = dropout_layer(V)
        V = Dense(256, activation="relu")(V)
        #V = dropout_layer(V)
        V = Dense(64, activation="relu")(V)
        #V = dropout_layer(V)
        #value = Dense(1, activation=None)(V)'''

        ### LSTM Model
        '''X = LSTM(512, return_sequences=True, dropout=.2)(X_input)
        X = LSTM(256, return_sequences=True)(X)
        X = BatchNormalization()(X)
        X = Dense(256, activation="relu")(X)
        X = LSTM(1)(X)'''

        ### CNN Model
        #dropout_layer = Dropout(.2)
        V = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
        #V = Dropout(.1)(V)
        V = MaxPooling1D(pool_size=2)(V)
        #V = Conv1D(filters=64, kernel_size=7, padding="same", activation="tanh")(X_input) #(V)
        #V = MaxPooling1D(pool_size=2)(V)
        #V = dropout_layer(V)
        #V = Conv1D(filters=32, kernel_size=5, padding="same", activation="tanh")(V)
        #V = MaxPooling1D(pool_size=2)(V)
        V = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(V)
        #V = Dropout(.1)(V)
        V = MaxPooling1D(pool_size=2)(V)
        #V = dropout_layer(V)
        V = Flatten()(V)

        ###

        value = Dense(1, activation=None)(V)
        self.Critic = Model(inputs=X_input, outputs = value) # value --> X
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(learning_rate=learning_rate))

        #######

        ## Actor model
        ### Dense Model
        '''A = Dense(512, activation="relu")(X)
        A = Dense(256, activation="relu")(A)
        A = Dense(64, activation="relu")(A)'''

        ### Dense Model + DropOut
        '''dropout_layer = Dropout(.2)
        #A = dropout_layer(X)
        A = Dense(512, activation="relu")(X)
        #A = dropout_layer(A)
        A = Dense(512, activation="relu")(A)
        #A = dropout_layer(A)
        A = Dense(256, activation="relu")(A)
        #A = dropout_layer(A)
        A = Dense(64, activation="relu")(A)
        #A = dropout_layer(A)'''

        ### LSTM Model
        #TODO : Adding Dropout
        '''X = LSTM(512, return_sequences=True)(X_input)
        X = LSTM(256, return_sequences=True)(X)
        X = Dense(256, activation="relu")(X)
        X = LSTM(1)(X)'''

        ### CNN Model
        dropout_layer = Dropout(.1)
        A = Conv1D(filters=64, kernel_size=6, padding="same", activation="tanh")(X_input)
        #A = Dropout(.2)(A)
        A = MaxPooling1D(pool_size=2)(A)
        #A = Conv1D(filters=64, kernel_size=7, padding="same", activation="tanh")(X_input) #(A)
        #A = MaxPooling1D(pool_size=2)(A)
        #A = dropout_layer(A)
        #A = Conv1D(filters=32, kernel_size=5, padding="same", activation="tanh")(A)
        #A = MaxPooling1D(pool_size=2)(A)
        A = Conv1D(filters=32, kernel_size=3, padding="same", activation="tanh")(A)
        #A = Dropout(.1)(A)
        A = MaxPooling1D(pool_size=2)(A)
        #A = dropout_layer(A)
        A = Flatten()(A)

        output = Dense(self.action_space, activation="softmax")(A) # A --> X

        self.Actor = Model(inputs = X_input, outputs = output) 
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
        #print(self.Actor.summary())

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])

        
class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space

        X = Flatten(input_shape=input_shape)(X_input)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(64, activation="relu")(X)
        output = Dense(self.action_space, activation="softmax")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss, optimizer=optimizer(lr=lr))
        #print(self.Actor.summary)

    def ppo_loss(self, y_true, y_pred):
        # Defined in https://arxiv.org/abs/1707.06347
        advantages, prediction_picks, actions = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space:]
        LOSS_CLIPPING = 0.2
        ENTROPY_LOSS = 0.001
        
        prob = actions * y_pred
        old_prob = actions * prediction_picks

        prob = K.clip(prob, 1e-10, 1.0)
        old_prob = K.clip(old_prob, 1e-10, 1.0)

        ratio = K.exp(K.log(prob) - K.log(old_prob))
        
        p1 = ratio * advantages
        p2 = K.clip(ratio, min_value=1 - LOSS_CLIPPING, max_value=1 + LOSS_CLIPPING) * advantages

        actor_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = actor_loss - entropy

        return total_loss

    def actor_predict(self, state):
        return self.Actor.predict(state)

class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)

        V = Flatten(input_shape=input_shape)(X_input)
        V = Dense(512, activation="relu")(V)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss

    def critic_predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
