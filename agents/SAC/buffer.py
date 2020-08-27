import numpy as np
import tensorflow as tf

class Buffer:
    def __init__(self, num_states,num_actions,buffer_capacity=100000, batch_size=64):

        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        BUFSIZE_STATE = (self.buffer_capacity,) + num_states
        self.state_buffer = np.zeros(BUFSIZE_STATE,dtype=np.float32)
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions),dtype=np.float32)
        self.reward_buffer = np.zeros((self.buffer_capacity, 1),dtype=np.float32)
        self.next_state_buffer = np.zeros(BUFSIZE_STATE,dtype=np.float32)
        self.done_buffer = np.zeros((self.buffer_capacity, 1),dtype=np.float32)

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]
        self.done_buffer[index] = obs_tuple[4]
        self.buffer_counter += 1

    # We compute the loss and update parameters
    def sample_batch(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])
        done_batch = np.array(self.done_buffer[batch_indices])
        
        reward_batch = tf.squeeze(reward_batch)
        done_batch = np.squeeze(done_batch)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch