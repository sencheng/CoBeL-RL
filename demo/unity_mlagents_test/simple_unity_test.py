import tensorboard as tb
from keras import backend
from agents.dqn_agents import RobotDQNAgent
from agents.dqn_agents import DDPGAgentBaseline
from interfaces.oai_gym_interface import unity_wrapper
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
    print(trial)
    # end learning after n trials
    if trial == rl_agent.trialNumber - 1:
        rl_agent.agent.step = rl_agent.maxSteps + 1  # force a max_steps exit


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


def single_run(environment_filename, n_train=1):
    """
    :param environment_filename: full path to a Unity executable
    :param n_train: amount of RL steps
    :return:

    This method performs a single experimental run, i.e. one experiment. It has to be called by either a parallelization
    mechanism (without visual output), or by a direct call (in this case, visual output can be used).
    """

    # set random seed
    seed = 42  # 42 is used for good luck. If more luck is needed try 4, 20, or a combination. If absolutely nothing works, try 13. The extra bad luck will cause a buffer overflow and then we're in. Pardon the PEP.

    unity_gym = unity_wrapper(env_path=environment_filename, modules=None, withGUI=visualOutput,
                                            seed=seed, agent_action_type="discrete")

    rl_agent = RobotDQNAgent(interfaceOAI=unity_gym,
                            trialBeginFcn=trial_begin_callback,
                            trialEndFcn=trial_end_callback)

    ''' THIS NEED MORE WORK
    rl_agent = DDPGAgentBaseline(interfaceOAI=unity_gym,
                                 memoryCapacity=5000,
                                 trialBeginFcn=trial_begin_callback,
                                 trialEndFcn=trial_end_callback
                                 )
                                 '''

    # set the experimental parameters
    rl_agent.trialNumber = 1000

    # let the agent learn, with extremely large number of allowed maximum steps
    rl_agent.train(n_train)

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
    #make your own agent adapted to the problem.
    project = get_cobel_rl_path()
    print('Testing environment 1')
    single_run(environment_filename=project+'/envs/win/Robot/UnityEnvironment', n_train=1000000)
    print('Testing concluded: No program breaking bugs detected.')
    print('Start tensorboard from unity_mlagents_test/logs/fit to see that the environments are learnable.')
