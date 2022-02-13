from distutils.sysconfig import customize_compiler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv1D, MaxPooling1D, LSTM, BatchNormalization, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from keras.layers.merge import concatenate
from tensorflow.keras import backend as K
from tensorflow.python.keras.backend import dropout
from copy import deepcopy
from time import sleep
from os.path import isfile
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass
class Base_CNN:
    def __init__(self, input_shapes, action_space, metalearner_action_space, learning_rate, optimizer, models = []):

        self.action_space = action_space
        self.metalearner_action_space = metalearner_action_space

        models_list = {
            'CNN_1': self.gen_cnn_1,
            'CNN_2': self.gen_cnn_2
        }

        self.models = []


        for i, model in enumerate(models):
            sleep(1)
            if model not in models_list:
                self.models.append({'model': self.load_custom_model(model), 'fit': True})                
            else:
                temp = models_list[model](input_shape = tuple(input_shapes[i]), output = action_space)
                # temp.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
                self.models.append({'model': temp, 'fit': False})

        # merged_layer = concatenate(axis=1)([model.output for model in compiled_models])
        # merged_layer = Concatenate(axis=-1)([model.output for model in self.models])
        final_input = Input(shape=action_space * len(models))
        layer_1 = Dense(10, activation='silu')(final_input)
        final_layer = Dense(action_space, activation='softmax')(layer_1)
        self.Base_Model = Model(inputs=final_input, outputs=final_layer)
        self.Base_Model.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
        # plot_model(self.Base_Model, to_file="metalearner.png", show_shapes=True)
    
    def gen_cnn_1(self, input_shape, output, optimizer=Adam, learning_rate=0.00005):
        X_input = Input(input_shape)
        CNN_1 = Conv1D(filters=256, kernel_size=9, padding="same", activation="relu")(X_input)
        CNN_1 = MaxPooling1D(pool_size=2, padding='same')(CNN_1)
        CNN_1 = Conv1D(filters=64, kernel_size=6, padding="same", activation="relu")(CNN_1)
        CNN_1 = MaxPooling1D(pool_size=2, padding='same')(CNN_1)
        CNN_1 = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(CNN_1)
        CNN_1 = MaxPooling1D(pool_size=2, padding='same')(CNN_1)
        CNN_1 = Flatten()(CNN_1)
        CNN_1_output = Dense(output, activation="softmax")(CNN_1)
        temp = Model(inputs = X_input, outputs = CNN_1_output)
        temp.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
        return temp


    def gen_cnn_2(self, input_shape, output, optimizer=Adam, learning_rate=0.00005):
        X_input = Input(input_shape)
        CNN_2 = Conv1D(filters=256, kernel_size=9, padding="same", activation="relu")(X_input)
        CNN_2 = MaxPooling1D(pool_size=2, padding='same')(CNN_2)
        CNN_2 = Conv1D(filters=64, kernel_size=6, padding="same", activation="relu")(CNN_2)
        CNN_2 = MaxPooling1D(pool_size=2, padding='same')(CNN_2)
        CNN_2 = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(CNN_2)
        CNN_2 = MaxPooling1D(pool_size=2, padding='same')(CNN_2)
        CNN_2 = Flatten()(CNN_2)
        CNN_2_output = Dense(output, activation="softmax")(CNN_2)
        temp = Model(inputs = X_input, outputs = CNN_2_output)
        temp.compile(loss=self.ppo_loss, optimizer=optimizer(learning_rate=learning_rate))
        return temp

    def load_custom_model(self, input_file):
        if not isfile(input_file):
            raise Exception(f'file {input_file} doesn\'t exists')
        return load_model(input_file, custom_objects={'ppo_loss': self.ppo_loss})

    def fit(self, inputs, target, epochs, verbose=0, shuffle=True, batch_size=64):
        inputs = np.array(inputs)
        for i, model in enumerate(self.models):
            # print(f'fittin model {i} of {len(self.models)}')
            if not model['fit']:
                model['model'].fit(inputs[i], target, epochs=epochs, verbose=verbose, shuffle=shuffle, batch_size=batch_size)

        temp_output = np.empty([inputs.shape[-3], 0])
        for i,model in enumerate(self.models):
            temp_output = np.append(temp_output, model['model'].predict(inputs[i]), axis=1)
        # target = target[:2000,:2]
        return self.Base_Model.fit(temp_output, target, epochs=epochs, verbose=verbose, shuffle=shuffle, batch_size=batch_size)


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

        agent_loss = -K.mean(K.minimum(p1, p2))

        entropy = -(y_pred * K.log(y_pred + 1e-10))
        entropy = ENTROPY_LOSS * K.mean(entropy)
        
        total_loss = agent_loss - entropy

        return total_loss
    

    def predict(self, states):
        temp_output = np.array([[] for _ in range(len(self.models))])
        for i in self.models:
            # temp_output = np.column_stack((temp_output, i['model'].predict(states)))
            temp_output = np.append(temp_output, i['model'].predict(states), axis=1)
        return self.Base_Model.predict(temp_output)
        # return self.Base_Model.predict(list(states))


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
        V = Conv1D(filters=256, kernel_size=9, padding="same", activation="relu")(X_input)
        V = MaxPooling1D(pool_size=2, padding='same')(V)
        V = Conv1D(filters=64, kernel_size=6, padding="same", activation="relu")(V)
        V = MaxPooling1D(pool_size=2, padding='same')(V)
        V = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(V)
        V = MaxPooling1D(pool_size=2, padding='same')(V)
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
        A = Conv1D(filters=256, kernel_size=9, padding="same", activation="relu")(X_input)
        A = MaxPooling1D(pool_size=2, padding='same')(A)
        A = Conv1D(filters=64, kernel_size=6, padding="same", activation="relu")(A)
        A = MaxPooling1D(pool_size=2, padding='same')(A)
        A = Conv1D(filters=32, kernel_size=3, padding="same", activation="relu")(A)
        A = MaxPooling1D(pool_size=2, padding='same')(A)
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

        # ____This is a Seperator not a comment____
        X = Flatten(input_shape=input_shape)(X_input)
        X = Dense(512, activation="relu")(X)
        X = Dense(256, activation="relu")(X)
        X = Dense(64, activation="relu")(X)
        output = Dense(self.action_space, activation="softmax")(X)
        # ____This is a Seperator not a comment____

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

        # ____This is a Seperator not a comment____
        V = Flatten(input_shape=input_shape)(X_input)
        V = Dense(512, activation="relu")(V)
        V = Dense(256, activation="relu")(V)
        V = Dense(64, activation="relu")(V)
        value = Dense(1, activation=None)(V)
        # ____This is a Seperator not a comment____

        self.Critic = Model(inputs=X_input, outputs = value)
        self.Critic.compile(loss=self.critic_PPO2_loss, optimizer=optimizer(lr=lr))

    def critic_PPO2_loss(self, y_true, y_pred):
        value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
        return value_loss
    
    def load(self, h5_file):
        self.Critic = load_model(h5_file, custom_objects={'critic_PPO2_loss': self.critic_PPO2_loss})

    def critic_predict(self, state):
        # return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
        # print(self.Critic.input)
        return self.Critic.predict(state)
