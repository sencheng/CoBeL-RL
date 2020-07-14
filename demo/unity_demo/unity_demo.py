import os
import keras
from keras import backend
from agents.dqn_agents import DQNAgentBaseline
from agents.modular_agents import ModularDDPGAgent as MDDPGAgent, ModularDQNAgent as MDQNAgent
from analysis.rl_monitoring.rl_performance_monitors import UnityPerformanceMonitor
from interfaces.oai_gym_interface import UnityInterface
from interfaces.oai_gym_interface import get_cobel_path, get_env_path


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # reduces the amount of debug messages from tensorflow.
backend.set_image_data_format(data_format='channels_last')
backend.set_image_dim_ordering('th')
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


def single_run(env_exec_path, scene_name=None, n_train=1):
    """
    :param env_exec_path:           full path to a Unity executable
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
    unity_env = UnityInterface(env_path=env_exec_path, scene_name=scene_name,
                               nb_max_episode_steps=100, decision_interval=10, agent_action_type="discrete",
                               performance_monitor=UnityPerformanceMonitor(update_period=1,
                                                                           reward_plot_viewbox=(-10, 10, 50),
                                                                           steps_plot_viewbox=(0, 100, 50)),
                               with_gui=True)

    # then you can set some experiment parameters
    # these are specific to the environment you've chosen. you can find the parameters for the examples here:
    # https://github.com/Unity-Technologies/ml-agents/blob/0.15.1/docs/Learning-Environment-Examples.md
    # below you can find the parameters for the custom envs

    """
    robot_maze parameters
    
    unity_env.env_configuration_channel.set_property("has_walls", 1)                # enable walls
    unity_env.env_configuration_channel.set_property("maze_algorithm", 0)           # just exterior walls
    unity_env.env_configuration_channel.set_property("size_x", 2)                   # set cell grid width
    unity_env.env_configuration_channel.set_property("size_y", 2)                   # set cell grid height
    unity_env.env_configuration_channel.set_property("random_target_pos", 0)        # disable target repositioning
    unity_env.env_configuration_channel.set_property("random_rotation_mode", 1)     # enable random robot spawn rotation
    unity_env.env_configuration_channel.set_property("max_velocity", 0)             # disable max agent velocity
    unity_env.env_configuration_channel.set_property("target_reached_radius", 20)   #
    unity_env.env_configuration_channel.set_property("target_visible", 1)           #
    """

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

    # create your agent with the unity processor
    rl_agent = DQNAgentBaseline(interfaceOAI=unity_env, processor=unity_env.processor)

    # train the agent
    rl_agent.train(n_train)

    # clear session
    backend.clear_session()
    unity_env.close()


if __name__ == "__main__":
    # TODO Make a loop and try out different hyperparameters.
    project = get_cobel_path()
    env_path = get_env_path()
    single_run(env_exec_path=env_path,
               scene_name="VisualRandomRobotMaze",
               n_train=1000)
