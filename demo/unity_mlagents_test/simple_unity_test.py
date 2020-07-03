import os
import datetime
import numpy as np
from pathlib import Path

import tensorflow as tf
from keras import backend
from agents.dqn_agents import ModularDQNAgentBaseline, sequential_memory_modul, sequential_model_modul

from interfaces.oai_gym_interface import UnityInterface
from analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy

import pyqtgraph as pg

visualOutput = True


def reward_callback(*args, **kwargs):
    """
    ATTENTION: This function is deprecated.
    These changes should be encoded in the Academy object of the environment, and triggered via a side channel, or in
    the Agent definition inside Unity.

    :return: None
    """

    raise NotImplementedError('This function is deprecated. These changes should either be encoded in the Academy '
                              'object of the environment, and triggered via a side channel, or in the Agent definition'
                              'inside Unity.')


def trial_begin_callback(trial, rl_agent):
    """
    This is a callback function that is called in the beginning of each trial. Here, experimental behavior can be
    defined (ABA renewal and the like).
    :param trial: the number of the finished trial
    :param rl_agent: the employed reinforcement learning agent
    :return: None
    """
    pass
    # print("Episode begin")


def trial_end_callback(trial, rl_agent, logs):
    """
    This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation
    can be introduced.
    :param trial: the number of the finished trial
    :param rl_agent: the employed reinforcement learning agent
    :param logs: output of the reinforcement learning subsystem
    :return:
    """
    print("Episode end", logs)


def single_run(environment_filename, scene_name=None, n_train=1):
    """
    :param environment_filename:    full path to a Unity executable
    :param scene_name:              the name of the scene to be loaded
    :param n_train:                 total number of rl steps. Note that it's affected by the action repetition
    :return:

    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization
    mechanism (without visual output), or by a direct call (in this case, visual output can be used).

    In this case it acts as a tutorial for using the UnityInterface
    """

    # first you create your environment, you can check detailed parameter descriptions in the constructor docstring.
    #
    # note that the decision interval is set to 10.
    # this means that cobel only observers and acts in every 10th simulation step. this is aka frame skipping.
    # it increases the performance and is helpful when training envs where the consequence of an action can only
    # be observed after some time.
    unity_env = UnityInterface(env_path=environment_filename, scene_name=scene_name, modules=None, with_gui=True,
                               seed=42, agent_action_type="discrete", nb_max_episode_steps=1000, decision_interval=15,
                               performance_monitor=UnityPerformanceMonitor(update_period=1))

    # then you can set some experiment parameters
    # these are specific to the environment you've chosen. you can find the parameters for the examples here:
    # https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Examples.md
    # below you can find the parameters for the custom envs

    """
    robot_maze parameters
    """

    unity_env.env_configuration_channel.set_property("has_walls", 0)                 # enable walls
    unity_env.env_configuration_channel.set_property("maze_algorithm", 1)           # Random DFS Maze
    unity_env.env_configuration_channel.set_property("size_x", 2)                   # set cell grid width
    unity_env.env_configuration_channel.set_property("size_y", 2)                   # set cell grid height
    unity_env.env_configuration_channel.set_property("random_target_pos", 0)        # disable target repositioning
    unity_env.env_configuration_channel.set_property("random_rotation_mode", 1)     # enable random robot spawn rotation
    unity_env.env_configuration_channel.set_property("max_velocity", 0)             # disable max agent velocity
    unity_env.env_configuration_channel.set_property("target_reached_radius", 25)


    """
    morris_water_maze parameters
    

    # scale the water pool size. default is 150x150 cm
    unity_env.env_configuration_channel.set_property("area_scale", 1)
    # set the platform position (1 = north, 2 = north east, ...)
    unity_env.env_configuration_channel.set_property("platform_direction", 1)
    # the scale factor of the platform. default is 10x10 cm
    unity_env.env_configuration_channel.set_property("platform_scale", 4)
    # whether of not the platform is visible to the rat agent
    unity_env.env_configuration_channel.set_property("platform_visible", 1)
    """

    # don't forget to reset your env after setting the experimental parameters to apply them
    unity_env._reset()

    # you can add some callbacks to the keras agent
    log_dir = Path("logs/fit/" + datetime.datetime.now().strftime("%Y.%m.%d-%H-%M-%S"))  # create OS-agnostic path
    log_dir = str(log_dir)                                                               # extract as string
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    # create your agent and use the env like any open ai gym
    rl_agent = ModularDQNAgentBaseline(oai_env=unity_env,
                                       policy=LinearAnnealedPolicy(EpsGreedyQPolicy(), "eps", 1., 0.1, 0.05, 50000),
                                       nb_steps_warmup=10000,
                                       create_memory_fcn=sequential_memory_modul(limit=50000),
                                       create_model_fcn=sequential_model_modul(nb_units=192, nb_layers=4),
                                       batch_size=64,
                                       action_repetition=1, train_interval=1, memory_window=1, memory_interval=1,
                                       trial_begin_fcn=trial_begin_callback, trial_end_fcn=trial_end_callback,
                                       other_callbacks=[tensorboard_callback])

    # train the agent
    rl_agent.train(n_train)

    # save the weights, if you like.
    rl_agent.save(get_cobel_rl_path()+"/models/test.h5")

    # clear session
    backend.clear_session()
    unity_env.close()


def get_cobel_rl_path():
    """
    returns the cobel project path
    """
    paths = os.environ['PYTHONPATH'].split(os.pathsep)
    path = None
    for p in paths:
        if 'CoBeL-RL' in p:
            full_path = p
            base_folder = full_path.split(sep='CoBeL-RL')[0]
            path = base_folder + 'CoBeL-RL'
            break
    return path


if __name__ == "__main__":
    # TODO Make a loop and try out different hyperparameters.
    project = get_cobel_rl_path()
    print('Testing environment 1')
    single_run(environment_filename=project + '/envs/lin/unity_env',
               scene_name="PushBlock",
               n_train=500000)
    print('Start tensorboard from unity_ml-agents_test/logs/fit to see that the environments are learnable.')
    pg.QtGui.QApplication.exec_()
