import tensorboard as tb
import tensorflow as tf
from keras import backend
from agents.dqn_agents import RobotDQNAgent
from agents.dqn_agents import DDPGAgentBaseline
from interfaces.oai_gym_interface import unity_wrapper
from mlagents_envs.side_channel.float_properties_channel import FloatPropertiesChannel
import os

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
    #print("Episode begin")

def trial_end_callback(trial, rl_agent, logs):
    """
    This is a callback routine that is called when a single trial ends. Here, functionality for performance evaluation
    can be introduced.
    :param trial: the number of the finished trial
    :param rl_agent: the employed reinforcement learning agent
    :param logs: output of the reinforcement learning subsystem
    :return:
    """
    pass
    #print("Episode end")


def single_run(environment_filename, n_train=1):
    """
    :param environment_filename: full path to a Unity executable
    :param n_train: amount of RL steps
    :return:

    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization
    mechanism (without visual output), or by a direct call (in this case, visual output can be used).
    """
    tf.global_variables_initializer

    # set random seed
    seed = 42  # 42 is used for good luck. If more luck is needed try 4, 20, or a combination. If absolutely nothing works, try 13. The extra bad luck will cause a buffer overflow and then we're in. Pardon the PEP.

    # setup a float side channel
    float_properties_channel = FloatPropertiesChannel()

    unity_gym = unity_wrapper(env_path=environment_filename, modules=None, withGUI=visualOutput,
                              seed=seed, agent_action_type="discrete", side_channels=[float_properties_channel])

    rl_agent = RobotDQNAgent(interfaceOAI=unity_gym, trialBeginFcn=trial_begin_callback, trialEndFcn=trial_end_callback)

    ''' THIS NEED MORE WORK
    rl_agent = DDPGAgentBaseline(interfaceOAI=unity_gym,
                                 memoryCapacity=5000,
                                 trialBeginFcn=trial_begin_callback,
                                 trialEndFcn=trial_end_callback
                                 )
                                 '''

    # set the experimental parameters
    float_properties_channel.set_property("nb_max_episode_steps", 3000) # max steps of the agent
    float_properties_channel.set_property("target_reached_radius", 10)  # distance at which the target is considered reached
    float_properties_channel.set_property("target_spawn_distance", 30)  # spawn distance of the target
    float_properties_channel.set_property("add_to_spawn_angle", 45)     # the change of the spawn angle, when successfully reached
    unity_gym._reset()

    rl_agent.train(n_train)

    rl_agent.agent.model.save_weights("models/robot.h5")

    backend.clear_session()

    unity_gym.close()


def get_cobel_rl_path():
    paths = os.environ['PYTHONPATH'].split(';')
    path = None
    for p in paths:
        if 'CoBeL-RL' in p:
            full_path = p
            base_folder = full_path.split(sep='CoBeL-RL')[0]
            path = base_folder + 'CoBeL-RL'
            break

    return path


if __name__ == "__main__":
    #TODO Make a loop and try out different hyperparamters
    project = get_cobel_rl_path()
    print('Testing environment 1')
    single_run(environment_filename=project+'/envs/win/Robot/UnityEnvironment', n_train=1000000)
    print('Start tensorboard from unity_mlagents_test/logs/fit to see that the environments are learnable.')
