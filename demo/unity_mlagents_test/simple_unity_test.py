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


def demo_run(environment_filename, n_train=1):
    """
    :param environment_filename: full path to a Unity executable
    :param n_train: amount of RL steps
    :return:

    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization
    mechanism (without visual output), or by a direct call (in this case, visual output can be used).
    """

    # set random seed
    # 42 is used for good luck. If more luck is needed try 4, 20, or a combination. If absolutely nothing works,
    # try 13. The extra bad luck will cause a buffer overflow and then we're in. Pardon the PEP.
    seed = 42

    # create unity env
    unity_env = UnityInterface(env_path=environment_filename, modules=None, with_gui=True,
                               seed=seed, agent_action_type="discrete", nb_max_episode_steps=4000, decision_interval=10,
                               performance_monitor=UnityPerformanceMonitor(update_period=1))

    # set experiment parameters
    unity_env.env_configuration_channel.set_property("platform_scale", 4)

    # initial reset
    unity_env._reset()

    # tensorboard log callback
    log_dir = Path("logs/fit/" + datetime.datetime.now().strftime("%Y.%m.%d-%H-%M-%S"))  # create OS-agnostic path
    log_dir = str(log_dir)  # extract as string
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0)

    # create agent
    rl_agent = ModularDQNAgentBaseline(oai_env=unity_env,
                                       policy=LinearAnnealedPolicy(EpsGreedyQPolicy(), "eps", 1., 0.1, 0.05, 50000),
                                       nb_steps_warmup=10000, nb_max_episode_steps=4000,
                                       create_memory_fcn=sequential_memory_modul(limit=50000),
                                       create_model_fcn=sequential_model_modul(nb_units=64, nb_layers=3),
                                       action_repetition=1, train_interval=1, memory_window=1, memory_interval=1,
                                       batch_size=32,
                                       trial_begin_fcn=trial_begin_callback, trial_end_fcn=trial_end_callback,
                                       other_callbacks=[tensorboard_callback])

    rl_agent.train(n_train)

    rl_agent.save("robot_test")

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
    # TODO Make a loop and try out different hyperparamters
    project = get_cobel_rl_path()
    print('Testing environment 1')
    demo_run(environment_filename=project + '/envs/lin/examples/push_block/push_block', n_train=1000)
    print('Start tensorboard from unity_ml-agents_test/logs/fit to see that the environments are learnable.')
    pg.QtGui.QApplication.exec_()
