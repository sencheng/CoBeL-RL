import os
import time
import numpy as np
import tensorflow as tf
from interfaces.oai_gym_interface import UnityInterface, get_cobel_path, get_env_path
from random import randrange
from tensorflow.keras import backend

backend.set_image_data_format(data_format='channels_last')

from agents.DDPG.agent import DDPG_Agent
#from agents.SAC.agent import SACAgent

if __name__ == "__main__":
    project = get_cobel_path()
    environment_path = get_env_path()

    SCENE_NAME = "ContinuousRobotMaze"

    unity_env = UnityInterface(env_path=environment_path, scene_name=SCENE_NAME,
                               nb_max_episode_steps=10000000, decision_interval=4,
                               agent_action_type="continuous", use_gray_scale_images=True)

    #agent = SACAgent(unity_env)
    agent = DDPG_Agent(unity_env,4)
    agent.train()

    # clear session
    backend.clear_session()
    unity_env.close()
