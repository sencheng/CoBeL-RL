import tensorflow as tf
import numpy as np

from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from interfaces.oai_gym_interface import UnityInterface

from agents.DDPG.noise import OUActionNoise
from agents.DDPG.buffer import Buffer
from agents.utils.mish import Mish
from agents.utils.ringbuffer import RingBuffer

class DDPG_Agent:
    def __init__(self,env: UnityInterface,frameskip=4):
        self.u_env = env
        self.num_states = self.u_env.observation_space.shape + (frameskip,)
        self.num_actions = self.u_env.action_space.n
        self.upper_bound = self.u_env.action_space.high[0]
        self.lower_bound = self.u_env.action_space.low[0]

        self.tau = 0.005
        self.actor_lr  = 0.0001
        self.critic_lr = 0.0001
        self.gamma = 0.99
        self.hid_act = "tanh"
        self.mem_size = 50000
        self.batch_size = 256
        self.episodes = 10000
        self.eval = True
        
        self.std_dev = 0.2
        self.ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
        
        self.critic_optimizer = Adam(self.critic_lr)
        self.actor_optimizer = Adam(self.actor_lr)
    
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()
        
        #Load All Weights
        self.actor_model.load_weights('/home/wkst/Desktop/actor.h5')
        self.critic_model.load_weights('/home/wkst/Desktop/critic.h5')
        
        #Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())
        
        #Buffer
        self.buffer = Buffer(self.num_states,self.num_actions, self.mem_size, self.batch_size)
        self.ringbuffer = RingBuffer(4)
        
    
    def create_lenet5(self,input_layer):
        out = layers.Conv2D(12,5,1,padding="same", activation="tanh")(input_layer)
        out = layers.BatchNormalization()(out)
        out = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")(out)
        out = layers.Conv2D(32,5,1,padding="valid",activation="tanh")(out)
        out = layers.BatchNormalization()(out)
        out = layers.MaxPooling2D(pool_size=(2,2),strides=(2,2),padding="valid")(out)
        out = layers.Flatten()(out)
        out = layers.Dense(240,activation="tanh")(out)
        out = layers.BatchNormalization()(out)
        out = layers.Dense(168,activation="tanh")(out)
        out = layers.BatchNormalization()(out)
        return out

    def get_actor(self):
        last_init = tf.random_uniform_initializer(-0.003,0.003)
        state_input = layers.Input(shape=self.num_states)
        out = self.create_lenet5(state_input)

        outputs = layers.Dense(self.num_actions, activation="tanh", kernel_initializer=last_init)(out)
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(state_input, outputs)
        model.summary()
        return model

    def get_critic(self):
        state_input = layers.Input(shape=self.num_states)
        action_input = layers.Input(shape=(self.num_actions))

        out = self.create_lenet5(state_input)

        # Action as input
        action_out = layers.Dense(82, activation=self.hid_act, kernel_initializer=tf.keras.initializers.HeNormal)(action_input)
        action_out = layers.BatchNormalization()(action_out)

        #Concatenate Both Layers
        concat = layers.Concatenate()([out, action_out])
        concat_out = layers.Dense(64, activation=self.hid_act, kernel_initializer=tf.keras.initializers.HeNormal)(concat)

        outputs = layers.Dense(1)(concat_out)
        model = tf.keras.Model([state_input, action_input], outputs)
        model.summary()
        return model

    @tf.function
    def backprob_models(self,state_batch,action_batch,reward_batch,next_state_batch):
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch)
            y = reward_batch + self.gamma * self.target_critic([next_state_batch, target_actions])
            critic_value = self.critic_model([state_batch, action_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grad, self.critic_model.trainable_variables))
        
        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch)
            critic_value = self.critic_model([state_batch, actions])
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grad, self.actor_model.trainable_variables))

    def train_model(self):
        state_batch,action_batch,reward_batch,next_state_batch = self.buffer.sample_batch()
        self.backprob_models(state_batch,action_batch,reward_batch,next_state_batch)
        
    def update_target(self):
        new_weights = []
        target_variables = self.target_critic.weights
        for i, variable in enumerate(self.critic_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))
        self.target_critic.set_weights(new_weights)

        new_weights = []
        target_variables = self.target_actor.weights
        for i, variable in enumerate(self.actor_model.weights):
            new_weights.append(variable * self.tau + target_variables[i] * (1 - self.tau))
        self.target_actor.set_weights(new_weights)

    def sample_action(self,state):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise()
        
        if self.eval == True:
            sampled_actions = sampled_actions.numpy()
        else:
            sampled_actions = sampled_actions.numpy() + noise

        #Action Bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)
        return [np.squeeze(legal_action)]
    
    def train(self):
        for ep in range(self.episodes):
            score = 0.0
            prev_state = self.u_env._reset()

            self.ringbuffer.insert_obs(prev_state[0][:,:,0])
            self.ringbuffer.insert_obs(prev_state[0][:,:,1])
            self.ringbuffer.insert_obs(prev_state[0][:,:,2])
            self.ringbuffer.insert_obs(prev_state[0][:,:,2])
            prev_state = self.ringbuffer.generate_arr()
            
            while True:
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)
                action = np.array(self.sample_action(tf_prev_state))
                state, reward, done, info = self.u_env._step(action)#
                if reward > 0.0:
                    score += reward
                
                #Get Each third Frame as workaround
                if state[0].shape == self.num_states:
                    self.ringbuffer.insert_obs(state[0][:,:,2])
                else:
                    self.ringbuffer.insert_obs(state[0][0][:,:,2])
                
                state = self.ringbuffer.generate_arr()
                
                if self.eval == False:
                    self.buffer.record((prev_state, action[0], reward, state))
                    self.train_model()
                    self.update_target()

                prev_state = state
                if done:
                    break
            print("ep" , ep, ": ", score)
            if self.eval == False:
                if score >= 3:
                    print("solved!")
                    self.actor_model.save_weights("actor.h5")
                    self.critic_model.save_weights("critic.h5")
                if score >= 5:
                    print("final solve!")
                    self.actor_model.save_weights("actor_f.h5")
                    self.critic_model.save_weights("critic_f.h5")