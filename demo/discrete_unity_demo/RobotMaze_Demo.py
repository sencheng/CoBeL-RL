import os
import time
from interfaces.oai_gym_interface import UnityInterface, get_cobel_path, get_env_path
from analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
from random import randrange
from keras import backend
import numpy as np
from PIL import Image

from agents.RDQN.agent import RDQNAgent
from agents.A2C_TF2.a2c_disc import A2CAgent

# set some python environment properties
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
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
                               nb_max_episode_steps=1000, decision_interval=1,
                               agent_action_type="discrete", use_gray_scale_images=False)

    
    #agent = RDQNAgent(unity_env,1000)
    agent = A2CAgent(unity_env)
    agent.train(1000)

    # clear session
    backend.clear_session()
    unity_env.close()
