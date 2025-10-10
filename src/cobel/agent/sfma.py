# basic imports
import copy
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from .agent import Callbacks
from ..memory.sfma import SFMAMemory, Experience
from ..policy.policy import Policy
from ..interface.interface import Interface
# typing
from numpy.typing import NDArray, ArrayLike
from .agent import CallbackDict, Logs


class SFMA(Agent):
    """
    Implementation of a Dyna-Q agent using the Spatial Structure and Frequency-weighted
    Memory Access (SFMA) memory module. The Q-function is represented as a static table.

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    memory : SFMAMemory
        The agent's memory module.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    learning_rate : float, default=0.99
        The agent's learning rate.
    gamma : float, default=0.99
        The agent's discount factor.
    custom_callbacks : CallbackDict or None, optional
        The custom callbacks defined by the user.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

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
    M : SFMAMemory
        The memory module used by the Dyna-Q agent for storing
        environmental transitions.
    action_mask : NDArray
        A boolean 2D NumPy array of shape (`observation_space.n`, `action_space.n`)
        which represents an action mask that can be applied during action selection.
    mask_actions : bool
        A flag indicating whether action should be masked
        as defined in `action_mask`. Set to False by default.
    nb_replay : int
        The number of replays that are generated per trial. Defaults to 1.
    random : bool
        A flag indicating whether random replays should be generated.
        False by default.
    dynamic : bool
        A flag indicating whether the replay mode is dynamically
        determined based on the recent prediction error history.
        False by default.
    offline : bool
        A flag indicating whether the agent only learns from replay.
        False by default.
    start_replay : bool
        A flag indicating whether additional replays should be
        generated at trial begin. False by default.
    td : float
        Stores the TD-error for dynamic replay mode.
    rng : numpy.random.Generator
        A random number generator instance used for
        probablistic replay mode selection.

    Notes
    -----
    This agent only supports gym.spaces.Discrete for
    `observation_space` and `action_space`.

    Examples
    --------

    Here we initialize the SFMA agent for a discrete
    environment with 16 states and 4 actions. ::

        >>> import gymnasium as gym
        >>> from cobel.agent import SFMA
        >>> from cobel.memory import SFMAMemory
        >>> from cobel.policy import EpsilonGreedy
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> from cobel.memory.utils.metrics import DR
        >>> gridworld = make_open_field(4, 4, 0, 1)
        >>> policy = EpsilonGreedy(0.1)
        >>> metric = DR(4, 4, gridworld['sas'], 0.9, [])
        >>> memory = SFMAMemory(DR, 16, 4)
        >>> agent = SFMA(gym.spaces.Discrete(16),
        ...              gym.spaces.Discrete(4), policy, memory)

    """

    class CallbacksSFMA(Callbacks):
        """
        Callback class of the SFMA agent. Used for visualization and scenario control.

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

        def on_replay_begin(self, logs: Logs) -> Logs:
            """
            The following function is called whenever experiences are replayed,
            and executes callbacks defined by the user.

            Parameters
            ----------
            logs : Logs
                The replay log dictionary.

            Returns
            -------
            logs : Logs
                The updated replay log dictionary.
            """
            replay_logs: dict = copy.copy(logs)
            logs['agent'] = self.agent
            if 'on_replay_begin' in self.custom_callbacks:
                for callback in self.custom_callbacks['on_replay_begin']:
                    callback_logs = callback(logs)
                    if type(callback_logs) is dict:
                        replay_logs.update(callback_logs)

            return replay_logs

        def on_replay_end(self, logs: Logs) -> Logs:
            """
            The following function is called whenever experiences are replayed,
            and executes callbacks defined by the user.

            Parameters
            ----------
            logs : Logs
                The replay log dictionary.

            Returns
            -------
            logs : Logs
                The updated replay log dictionary.
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
        memory: SFMAMemory,
        policy_test: None | Policy = None,
        learning_rate: float = 0.99,
        gamma: float = 0.99,
        custom_callbacks: None | CallbackDict = None,
        rng: None | np.random.Generator = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Discrete, (
            'SFMA requires a discrete observation space!'
        )
        assert type(action_space) is gym.spaces.Discrete, (
            'SFMA requires a discrete action space!'
        )
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        self.rng = np.random.default_rng() if rng is None else rng
        self.callbacks = self.CallbacksSFMA(self, custom_callbacks)
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((observation_space.n, action_space.n))
        self.M = memory
        self.action_mask: NDArray = np.ones(
            (observation_space.n, action_space.n)
        ).astype(bool)
        self.mask_actions: bool = False
        # replays per trial
        self.nb_replays = 1
        # if true, random replay batches are sampled
        self.random: bool = False
        # if true, the replay mode is determined by the cumulative TD-error
        self.dynamic: bool = False
        # if true, the agent learns only with experience replay
        self.offline: bool = False
        # if true, a replay trace is generated at the start of each trial
        self.start_replay: bool = False
        # stores TD-error for dynamic mode
        self.td: float = 0.0

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
        assert type(self.callbacks) is self.CallbacksSFMA
        for trial in range(trials):
            last: None | int = None
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                    'replay_mode': self.M.mode,
                }
            )
            # reset environment
            state, _ = interface.reset()
            assert type(state) is int
            if self.start_replay:
                logs = self.callbacks.on_replay_begin(logs)
                logs['replay'] = self.M.replay(batch_size, state)
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
                self.M.store(experience)
                experience = self.update_q(experience)
                # update current state
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    last = next_state
                    break
            self.current_trial += 1
            # log steps
            logs['steps'] = step
            # perform experience replay
            if not no_replay:
                # determine replay mode if modes are chosen dynamically
                if self.dynamic:
                    p_mode = 1 / (1 + np.exp(-(self.td * 5 - 2)))
                    logs['replay_mode'] = ['reverse', 'default'][
                        self.rng.choice(np.arange(2), p=np.array([p_mode, 1 - p_mode]))
                    ]
                    self.M.mode = logs['replay_mode']
                    self.td = 0.0
                # replay
                for _ in range(self.nb_replays):
                    logs = self.callbacks.on_replay_begin(logs)
                    logs['replay'] = self.replay(batch_size, last)
                    logs = self.callbacks.on_replay_end(logs)
                self.M.T.fill(0)
            # callback
            logs = self.callbacks.on_trial_end(logs)
            if self.stop:
                break

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
        assert type(self.callbacks) is self.CallbacksSFMA
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                    'replay_mode': self.M.mode,
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
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            # log steps
            logs['steps'] = step
            # callback
            logs = self.callbacks.on_trial_end(logs)
            if self.stop:
                break

    def replay(self, batch_size: int, state: None | int = None) -> list[Experience]:
        """
        This function replays experiences to update the Q-function.

        Parameters
        ----------
        batch_size : int
            The number of experiences that will be replayed.
        state : int or None, optional
            The state at which replay should be initiated.

        Returns
        -------
        replay_batch : list of dict
            The batch of replayed experiences.
        """
        # sample batch of experiences
        replay_batch = []
        if self.random:
            mask = np.ones(np.prod(self.Q.shape))
            if self.mask_actions:
                mask = np.copy(self.action_mask).flatten(order='F')
            replay_batch = self.M.retrieve_random_batch(batch_size, mask)
        else:
            replay_batch = self.M.replay(batch_size, state)
        # update the Q-function with each experience
        for experience in replay_batch:
            self.update_q(experience)

        return replay_batch

    def update_q(self, experience: Experience, no_update: bool = False) -> Experience:
        """
        This function updates the Q-function with a given experience.

        Parameters
        ----------
        experience : dict
            A dictionary containing the experience tuple.
        no_update : bool, default=False
            If true, the Q-function is not updated.

        Returns
        -------
        experience : dict
            A dictionary containing the experience tuple and the TD-error.
        """
        # make mask
        mask = np.arange(self.Q.shape[1])
        if self.mask_actions:
            mask = self.action_mask[experience['next_state']]
        # compute TD-error
        td = experience['reward']
        td += (
            self.gamma
            * experience['terminal']
            * np.amax(self.Q[experience['next_state']][mask])
        )
        td -= self.Q[experience['state']][experience['action']]
        experience['td'] = td
        # update Q-function with TD-error
        if not no_update:
            self.Q[experience['state']][experience['action']] += self.learning_rate * td
        # if self.dynamic_mode:
        self.td += np.abs(td)

        return experience

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
