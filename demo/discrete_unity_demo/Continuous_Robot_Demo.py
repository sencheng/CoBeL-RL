import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)sess = tf.Session(config=config)set_session(sess)  # set this TensorFlow session as the default session for Keras

#tf.compat.v1.disable_eager_execution()

import os
import time
from interfaces.oai_gym_interface import UnityInterface, get_cobel_path, get_env_path
from analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
from random import randrange
from tensorflow.keras import backend
import numpy as np
from PIL import Image

#from agents.A2C_TF2.a2c_cont import A2CAgent
#from agents.SAC.agent import SACAgent
from agents.DDPG.agent import DDPG_Agent

# set some python environment properties
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
visualOutput = True
backend.set_image_data_format(data_format='channels_last')

def ProcessImage(observation):
    observation = observation.squeeze() * 255
    observation = observation[:,:,0]
    observation = np.array(observation, dtype=np.uint8)
    img = Image.fromarray(observation)
    img.save("test.png")

if __name__ == "__main__":
    project = get_cobel_path()
    environment_path = get_env_path()

    SCENE_NAME = "ContinuousRobotMaze"

    unity_env = UnityInterface(env_path=environment_path, scene_name=SCENE_NAME,
                               nb_max_episode_steps=10000000, decision_interval=4,
                               agent_action_type="continuous", use_gray_scale_images=True)

    
    #agent = A2CAgent(unity_env)
    #agent = SACAgent(unity_env)
    #agent.train(10000000)
    
    with tf.Session() as sess:
        with tf.device("/cpu:0"):
            agent = DDPG_Agent(sess,unity_env)
            agent.train()

    # while (True):
    #     in_1 = input("Input Forward(1)/Backward(-1)")
    #     in_2 = input("Input Right(1)/Left(-1)")
    #     observation, reward, done, info = unity_env._step(np.array([[float(in_1),float(in_2)]]))
    #     print(reward)
        #ProcessImage(np.array(observation))

    # clear session
    backend.clear_session()
    unity_env.close()
