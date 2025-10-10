# basic imports
import copy
import numpy as np
import gymnasium as gym
# framework imports
from ..memory.pma import PMAMemory, Experience
from .agent import Agent, Callbacks, CallbackDict, Logs
from ..policy.policy import Policy
from ..interface.interface import Interface
# typing
from numpy.typing import NDArray, ArrayLike


class PMA(Agent):
    """
    Implementation of a Dyna-Q agent using the Prioritized Memory Access (PMA)
    method described by Mattar & Daw (2018): https://doi.org/10.1038/s41593-018-0232-z

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    memory : PMAMemory
        The agent's memory module.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    learning_rate : float, default=0.9
        The agent's learning rate.
    gamma : float, default=0.99
        The agent's discount factor.
    custom_callbacks : CallbackDict or None, optional
        The custom callbacks defined by the user.

    Attributes
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
        If none was provided in `policy_test` then `policy` will be used here as well.
    gamma : float
        The agent's discount factor.
    learning_rate : float
        The agent's learning rate.
    Q : NDArray
        A 2D NumPy array of shape (`observation_space.n`, `action_space.n`)
        which represents the agent's Q-function.
        The Q-function is initialized to all zeros.
    M : PMAMemory
        The memory module used by the Dyna-Q agent for storing
        environmental transitions.
    action_mask : NDArray
        A boolean 2D NumPy array of shape (`observation_space.n`, `action_space.n`)
        which represents an action mask that can be applied during action selection.
    mask_actions : bool
        A flag indicating whether action should be masked
        as defined in `action_mask`. Set to False by default.

    Notes
    -----
    This agent only supports gym.spaces.Discrete for
    `observation_space` and `action_space`.

    Examples
    --------

    Here we initialize the PMA agent for a discrete
    environment with 16 states and 4 actions. ::

        >>> import gymnasium as gym
        >>> from cobel.agent import PMA
        >>> from cobel.memory import PMAMemory
        >>> from cobel.policy import EpsilonGreedy
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> gridworld = make_open_field(4, 4, 0, 1)
        >>> policy = EpsilonGreedy(0.1)
        >>> memory = PMAMemory(gridworld['sas'], policy)
        >>> agent = PMA(gym.spaces.Discrete(16),
        ...             gym.spaces.Discrete(4), policy, memory)

    """

    class CallbacksPMA(Callbacks):
        """
        Callback class of the PMA agent. Used for visualization and scenario control.

        Parameters
        ----------
        agent : Agent
            Reference to the RL agent.
        custom_callbacks : CallbackDict or None, optional
            The custom callbacks defined by the user.

        """

        def __init__(
            self, agent: Agent, custom_callbacks: None | CallbackDict = None
        ) -> None:
            super().__init__(agent, custom_callbacks)

        def on_replay_end(self, logs: Logs) -> Logs:
            """
            This function is called on the end of replay,
            and executes callbacks defined by the user.

            Parameters
            ----------
            logs : Logs
                The replay log.

            Returns
            -------
            logs : Logs
                The replay log.
            """
            replay_logs: dict = copy.copy(logs)
            logs['agent'] = self.agent
            if 'on_replay_end' in self.custom_callbacks:
                for callback in self.custom_callbacks['on_replay_end']:
                    callback_logs = callback(logs)
                    if type(callback_logs) is dict:
                        replay_logs.update(callback_logs)

            return replay_logs

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        memory: PMAMemory,
        policy_test: None | Policy = None,
        learning_rate: float = 0.9,
        gamma: float = 0.99,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Discrete, (
            'PMA requires a discrete observation space!'
        )
        assert type(action_space) is gym.spaces.Discrete, (
            'PMA requires a discrete action space!'
        )
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        self.callbacks = self.CallbacksPMA(self, custom_callbacks)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((observation_space.n, action_space.n))
        self.M = memory
        self.action_mask: NDArray = np.ones(
            (observation_space.n, action_space.n)
        ).astype(bool)
        self.mask_actions: bool = False

    def train(
        self,
        interface: gym.Env | Interface,
        trials: int,
        steps: int,
        batch_size: int = 32,
        no_replay: bool = False,
    ) -> None:
        """
        This function is called to train the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is trained.
        steps : int
            The maximum number of steps per trial.
        batch_size : int, default=32
            The number of experiences that will be replayed.
        no_replay : bool, default=False
            If true, experiences are not replayed.
        """
        assert type(self.callbacks) is self.CallbacksPMA
        for trial in range(trials):
            last: None | int = None
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0,
                    'steps': 0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            # reset environment
            state, _ = interface.reset()
            assert type(state) is int
            # perform experience replay
            if not no_replay:
                logs['replay'], self.Q = self.M.replay(
                    self.Q,
                    self.action_mask if self.mask_actions else None,
                    batch_size,
                    state,
                )
                logs = self.callbacks.on_replay_end(logs)
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.policy.select_action(
                    self.Q[state],
                    self.action_mask[state] if self.mask_actions else None,
                )
                next_state, reward, end_trial, truncated, log = interface.step(action)
                assert type(next_state) is int
                # update Q-function and store experience
                experience: Experience = {
                    'state': state,
                    'action': int(action),
                    'reward': float(reward),
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                self.update_q([experience])
                self.M.store(experience)
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    last = next_state
                    break
            self.current_trial += 1
            logs['steps'], logs['step'] = step, step
            # perform experience replay
            if not no_replay:
                self.M.update_sr()
                logs['replay'], self.Q = self.M.replay(
                    self.Q,
                    self.action_mask if self.mask_actions else None,
                    batch_size,
                    last,
                )
                logs = self.callbacks.on_replay_end(logs)
            # callback
            logs = self.callbacks.on_trial_end(logs)

    def test(self, interface: gym.Env | Interface, trials: int, steps: int) -> None:
        """
        This function is called to test the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is trained.
        steps : int
            The maximum number of steps per trial.
        """
        assert type(self.callbacks) is self.CallbacksPMA
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0,
                    'steps': 0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            # reset environment
            state, _ = interface.reset()
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.policy.select_action(
                    self.Q[state],
                    self.action_mask[state] if self.mask_actions else None,
                )
                next_state, reward, end_trial, truncated, log = interface.step(action)
                assert type(next_state) is int
                # update Q-function and store experience
                experience: Experience = {
                    'state': state,
                    'action': int(action),
                    'reward': float(reward),
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                # update current state
                state = next_state
                # update cumulative reward
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'], logs['step'] = step, step
            # callback
            logs = self.callbacks.on_trial_end(logs)

    def update_q(self, update: list[Experience]) -> None:
        """
        This function updates the Q-function.

        Parameters
        ----------
        update : list of dict
            A list containing experiences for an n-step update.
        """
        # expected future value
        future_value = (
            np.amax(self.Q[update[-1]['next_state']]) * update[-1]['terminal']
        )
        for s, step in enumerate(update):
            # sum rewards over remaining trajectory
            r = 0.0
            intermediate_terminal = 1
            for following_steps in range(len(update) - s):
                # check for intermediate terminal transitions
                if (update[s + following_steps]['terminal'] == 0) and (
                    s != len(update) - 1
                ):
                    intermediate_terminal = 0
                    break
                r += update[s + following_steps]['reward'] * (
                    self.gamma**following_steps
                )
            # abort in case a terminal transition occurs within the n-step sequence
            if intermediate_terminal == 0:
                break
            # compute TD-error
            td = r + future_value * (self.gamma ** (following_steps + 1))
            td -= self.Q[step['state']][step['action']]
            # update Q-function with TD-error
            self.Q[step['state']][step['action']] += self.learning_rate * td

    def predict_on_batch(self, batch: ArrayLike) -> NDArray:
        """
        This function retrieves Q-values for a batch of states.

        Parameters
        ----------
        batch : ArrayLike
            The batch of states for which Q-values should be retrieved.

        Returns
        -------
        predictions : NDArray
            The batch of Q-value predictions.
        """
        return self.Q[np.array(batch).astype(int)]
