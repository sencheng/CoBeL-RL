import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from numpy.random import seed
tf.compat.v1.disable_eager_execution()
# seed(42)
# tf.random.set_seed(42)
tf.get_logger().setLevel('INFO')

import time
from interfaces.oai_gym_interface import UnityInterface, get_cobel_path, get_env_path
from random import randrange
from tensorflow.keras import backend
import numpy as np
from PIL import Image

#from agents.RDQN.agent import RDQNAgent as RDQN
from agents.A2C.a2c_disc import A2CAgent as A2C

# set some python environment properties
visualOutput = True
backend.set_image_data_format(data_format='channels_last')

def ProcessImage(observation):
    print(observation.shape)
    observation = observation.squeeze() * 255
    observation = np.array(observation, dtype=np.uint8)
    img = Image.fromarray(observation, 'RGBA')
    img.save("test.png")

if __name__ == "__main__":
    project = get_cobel_path()
    environment_path = get_env_path()

    SCENE_NAME = "DiscreteRobotMaze"

    unity_env = UnityInterface(env_path=environment_path, scene_name=SCENE_NAME,
                               nb_max_episode_steps=3000000, decision_interval=1,
                               agent_action_type="discrete", use_gray_scale_images=False)

    
    #agent = RDQN(unity_env,1000000)
    agent = A2C(unity_env)
    agent.train(1000000)
    
    # while (True):
    #     in_1 = input("Move")
    #     observation, reward, done, info = unity_env._step(np.array([[float(in_1)]]))
    #     ProcessImage(np.array(observation))

    
    # clear session
    backend.clear_session()
    unity_env.close()
