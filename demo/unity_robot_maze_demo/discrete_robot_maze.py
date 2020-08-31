import tensorflow as tf
import time
import numpy as np

tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('INFO')

from numpy.random import seed
from interfaces.oai_gym_interface import UnityInterface, get_cobel_path, get_env_path
from tensorflow.keras import backend

from agents.RDQN.agent import RDQNAgent as RDQN
#from agents.A2C.Discrete.a2c_disc import A2CAgent as A2C

backend.set_image_data_format(data_format='channels_last')

if __name__ == "__main__":
    project = get_cobel_path()
    environment_path = get_env_path()

    SCENE_NAME = "DiscreteRobotMaze"

    unity_env = UnityInterface(env_path=environment_path, scene_name=SCENE_NAME,
                               nb_max_episode_steps=3000000, decision_interval=1,
                               agent_action_type="discrete", use_gray_scale_images=False)

    
    agent = RDQN(unity_env,1000000)
    #agent = A2C(unity_env)
    agent.train(1000000)
    
    # while True:
    #     action = float(input())
    #     unity_env._step(np.array([[action]]))

    # clear session
    backend.clear_session()
    unity_env.close()
