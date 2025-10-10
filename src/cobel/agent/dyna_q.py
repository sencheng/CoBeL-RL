# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from ..memory.dyna_q import DynaQMemory, Experience
from ..policy.policy import Policy
from ..network.network import Network
from ..interface.interface import Interface
# typing
from numpy.typing import ArrayLike, NDArray
from .agent import CallbackDict


class DynaQ(Agent):
    """
    This class implements a Dyna-Q agent.
    The Q-function is represented as a static table.

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    learning_rate : float, default=0.99
        The agent's learning rate.
    gamma : float, default=0.99
        The agent's discount factor.
    memory : DynaQMemory or None, optional
        The agent's memory module.
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
    M : DynaQMemory
        The memory module used by the Dyna-Q agent for storing
        environmental transitions.
        If none was provided in `memory` an instance with default
        parameters will be created.
    action_mask : NDArray
        A boolean 2D NumPy array of shape (`observation_space.n`, `action_space.n`)
        which represents an action mask that can be applied during action selection.
    mask_actions : bool
        A flag indicating whether action should be masked
        as defined in `action_mask`. Set to False by default.
    episodic_replay : bool
        A flag indicating whether replay should only occurr
        at the end of a trial. Set to False by default.

    Notes
    -----
    This agent only supports gym.spaces.Discrete for `observation_space`
    and `action_space`.

    Examples
    --------

    Here we initialize the Dyna-Q agent for a discrete
    environment with 16 states and 4 actions. ::

        >>> import gymnasium as gym
        >>> from cobel.agent import DynaQ
        >>> from cobel.policy import EpsilonGreedy
        >>> agent = DynaQ(gym.spaces.Discrete(16),
        ...         gym.spaces.Discrete(4), EpsilonGreedy(0.1))

    Training and testing of the agent is done with the train
    and test methods, respectively.
    Both methods expect an RL interface and number of trials
    as input.::

        >>> from cobel.interface import Gridworld
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> env = Gridworld(make_open_field(4, 4, 0, 1))
        >>> agent.train(env, 100)
        >>> agent.test(env, 100)

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        policy_test: None | Policy = None,
        learning_rate: float = 0.99,
        gamma: float = 0.99,
        memory: None | DynaQMemory = None,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Discrete, (
            'DynaQ requires a discrete observation space!'
        )
        assert type(action_space) is gym.spaces.Discrete, (
            'DynaQ requires a discrete action space!'
        )
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q = np.zeros((observation_space.n, action_space.n))
        self.M: DynaQMemory
        if memory is None:
            self.M = DynaQMemory(int(observation_space.n), int(action_space.n))
        else:
            self.M = memory
        self.action_mask: NDArray = np.ones(
            (observation_space.n, action_space.n)
        ).astype(bool)
        self.mask_actions: bool = False
        self.episodic_replay: bool = False

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
            The batch size used for experience replay.
        no_replay : bool, default=False
            Flag indicating whether experience replay is disabled.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
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
                self.M.store(experience)
                experience = self.update_q(experience)
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(batch_size)
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(batch_size)
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
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
            The number of trials that the agent is tested.
        steps : int
            The maximum number of steps per trial.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            state, _ = interface.reset()
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.policy_test.select_action(
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
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def update_q(self, experience: Experience) -> Experience:
        """
        This function updates the Q-function for a given experience.

        Parameters
        ----------
        experience : Experience
            A dictionary containing the experience tuple.

        Returns
        -------
        experience : Experience
            A dictionary containing the experience tuple and the TD-error.
        """
        # compute TD-error
        td = experience['reward']
        td += (
            self.gamma
            * experience['terminal']
            * np.amax(self.Q[experience['next_state']])
        )
        td -= self.Q[experience['state']][experience['action']]
        experience['td'] = td
        # update Q-function
        self.Q[experience['state']][experience['action']] += self.learning_rate * td

        return experience

    def predict_on_batch(self, batch: ArrayLike) -> NDArray:
        """
        This function retrieves the Q-values for a batch of observations.

        Parameters
        ----------
        batch : ArrayLike
            The batch of observations for which Q-values should be retrieved.

        Returns
        -------
        predictions : NDArray
            The batch of Q-value predictions.
        """
        return self.Q[np.array(batch).astype(int)]

    def replay(self, batch_size: int) -> None:
        """
        This function retrieves a batch of experiences and updates the Q-function.

        Parameters
        ----------
        batch_size : int
            The number of experiences that will be retrieved.
        """
        batch = self.M.retrieve_batch(batch_size)
        for experience in batch:
            self.update_q(experience)


class DynaDQN(Agent):
    """
    This class combines the Dyna-Q model with the DQN algorithm.
    Gridworld states are mapped to predefined observations and fed into the network.

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    model : Network
        The network model used by the agent.
    observations : NDArray or None, optional
        The observations that will be mapped to gridworld states.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    gamma : float, default=0.99
        The agent's discount factor.
    memory : DynaQMemory or None, optional
        The agent's memory module.
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
    model_target : Network
        The target network which is used to compute target values.
    model_online : Network
        The online network which is used for behavior.
    observations : NDArray
        The observations that will be mapped to environmental states.
        If none were provided in `observations` then a one-hot
        encoding will be used.
    target_update : float
        Determines when and how `model_target` is updated.
        For values in the interval (0, 1) 'model_target' is blended
        with `model_online` using `target_update`.
        For values greater 1 `model_target` is updated with
        `model_online` every `target_update` steps.
    last_update : int
        Tracks the number steps since the last update of `model_target`.
    DDQN : bool
        A flag indicating whether Double DQN algorithm should be used
        when updating `model_online`.
    gamma : float
        The agent's discount factor.
    M : DynaQMemory
        The memory module used by the Dyna-Q agent for storing
        environmental transitions.
        If none was provided in `memory` an instance with default
        parameters will be created.
    action_mask : NDArray
        A boolean 2D NumPy array of shape (`observation_space.n`, `action_space.n`)
        which represents an action mask that can be applied during action selection.
    mask_actions : bool
        A flag indicating whether action should be masked
        as defined in `action_mask`. Set to False by default.
    episodic_replay : bool
        A flag indicating whether replay should only occurr
        at the end of a trial. Set to False by default.

    Notes
    -----
    This agent only supports gym.spaces.Discrete for `observation_space`
    and `action_space`.

    Examples
    --------

    Here we initialize the Dyna-DQN agent for a discrete
    environment with 16 states and 4 actions. ::

        >>> import torch.nn as nn
        >>> import gymnasium as gym
        >>> from collections import OrderedDict
        >>> from cobel.agent import DynaDQN
        >>> from cobel.policy import EpsilonGreedy
        >>> from cobel.network import TorchNetwork
        >>> layers = [('dense', nn.Linear(16, 64)),
        ...           ('act', nn.ReLU()),
        ...           ('out', nn.Linear(64, 4))]
        >>> network = nn.Sequential(OrderedDict(layers))
        >>> agent = DynaDQN(gym.spaces.Discrete(16), gym.spaces.Discrete(4),
        ...         TorchNetwork(network), EpsilonGreedy(0.1))

    Training and testing of the agent is done with the train
    and test methods, respectively.
    Both methods expect an RL interface and number of trials
    as input.::

        >>> from cobel.interface import Gridworld
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> env = Gridworld(make_open_field(4, 4, 0, 1))
        >>> agent.train(env, 100)
        >>> agent.test(env, 100)

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        model: Network,
        observations: None | NDArray = None,
        policy_test: None | Policy = None,
        gamma: float = 0.99,
        memory: None | DynaQMemory = None,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Discrete, (
            'DynaDQN requires a discrete observation space!'
        )
        assert type(action_space) is gym.spaces.Discrete, (
            'DynaDQN requires a discrete action space!'
        )
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        # build target and online models
        self.model_target = model
        self.model_online = self.model_target.clone()
        # generate one-encoding if no observations were provided
        assert type(self.observation_space) is gym.spaces.Discrete
        self.observations: NDArray
        if observations is None:
            self.observations = np.eye(int(self.observation_space.n))
        else:
            self.observations = observations
        self.target_update: float = 10**-2
        self.last_update: int = 0
        self.DDQN: bool = False
        self.gamma = gamma
        self.M: DynaQMemory
        if memory is None:
            self.M = DynaQMemory(int(observation_space.n), int(action_space.n))
        else:
            self.M = memory
        self.action_mask: NDArray = np.ones(
            (observation_space.n, action_space.n)
        ).astype(bool)
        self.mask_actions: bool = False
        self.episodic_replay: bool = False

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
            The batch size used for experience replay.
        no_replay : bool, default=False
            Flag indicating whether experience replay is disabled.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            state, _ = interface.reset()
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.policy.select_action(
                    self.retrieve_q(state),
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
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(batch_size)
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(batch_size)
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
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
            The number of trials that the agent is tested.
        steps : int
            The maximum number of steps per trial.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            state, _ = interface.reset()
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.policy_test.select_action(
                    self.retrieve_q(state),
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
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def retrieve_q(self, state: int) -> NDArray:
        """
        This function retrieves the Q-values for a given state.

        Parameters
        ----------
        state : int
            The state for which Q-values will be retrieved.

        Returns
        -------
        q_values : NDArray
            The Q-values.
        """
        return self.model_online.predict_on_batch(
            self.observations[state : (state + 1)]
        )[0]  # type: ignore

    def predict_on_batch(self, batch: ArrayLike) -> NDArray:
        """
        This function retrieves the Q-values for a batch of observations.

        Parameters
        ----------
        batch : ArrayLike
            The batch of observations for which Q-values should be retrieved.

        Returns
        -------
        predictions : NDArray
            The batch of Q-value predictions.
        """
        return self.model_online.predict_on_batch(
            self.observations[np.array(batch).astype(int)]
        )  # type: ignore

    def replay(self, batch_size: int) -> None:
        """
        This function retrieves a batch of experiences and updates the Q-function.

        Parameters
        ----------
        batch_size : int
            The number of experiences that will be retrieved.
        """
        states: list[NDArray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_states: list[NDArray] = []
        terminals: list[bool] = []
        for experience in self.M.retrieve_batch(batch_size):
            states.append(self.observations[experience['state']])
            actions.append(experience['action'])
            rewards.append(experience['reward'])
            next_states.append(self.observations[experience['next_state']])
            terminals.append(bool(experience['terminal']))
        # compute targets
        targets = self.model_online.predict_on_batch(np.array(states))
        bootstrap_values = self.model_target.predict_on_batch(np.array(next_states))
        bootstrap_actions: NDArray
        assert type(bootstrap_values) is np.ndarray and type(targets) is np.ndarray
        if self.DDQN:
            preds = self.model_online.predict_on_batch(np.array(next_states))
            assert type(preds) is NDArray
            bootstrap_actions = np.argmax(preds, axis=1)
        else:
            bootstrap_actions = np.argmax(bootstrap_values, axis=1)
        bootstrap_values = bootstrap_values[np.arange(batch_size), bootstrap_actions]
        assert type(bootstrap_values) is np.ndarray
        targets[np.arange(batch_size), actions] = (
            rewards + bootstrap_values * terminals * self.gamma
        )
        # update online model
        self.model_online.train_on_batch(np.array(states), targets)
        # update target model
        self.last_update += 1
        if self.target_update < 1.0:
            weights_target = np.array(self.model_target.get_weights(), dtype=object)
            weights_online = np.array(self.model_online.get_weights(), dtype=object)
            weights_target += self.target_update * (weights_online - weights_target)
            # self.model_target.set_weights(list(weights_target))
            self.model_target.set_weights(weights_target)  # type: ignore
        elif self.last_update == self.target_update:
            self.model_target.set_weights(self.model_online.get_weights())
            self.last_update = 0


class DynaDSR(Agent):
    """
    This class combines the Dyna-Q model with the DSR algorithm.
    Gridworld states are mapped to predefined observations and fed into the network.

    Parameters
    ----------
    observation_space : gym.Space
        The agent's observation space.
    action_space : gym.Space
        The agent's action space.
    policy : Policy
        The agent's action selection policy used during training.
    model_sr : Network
        The network model used by the agent to represent the successor representation.
    model_reward : Network
        The network model used by the agent to represent the reward function.
    observations : NDArray or None, optional
        The observations that will be mapped to gridworld states.
    policy_test : Policy or None, optional
        The agent's action selection policy used during testing.
    gamma : float, default=0.99
        The agent's discount factor.
    memory : DynaQMemory or None, optional
        The agent's memory module.
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
    model_target : dict of Network
        The target network which is used to compute target values.
    model_online : dict of Network
        The online network which is used for behavior.
    model_reward : Network
        The network which is used to represent the reward function.
    observations : NDArray
        The observations that will be mapped to environmental states.
        If none were provided in `observations` then a one-hot
        encoding will be used.
    target_update : float
        Determines when and how `model_target` is updated.
        For values in the interval (0, 1) 'model_target' is blended
        with `model_online` using `target_update`.
        For values greater 1 `model_target` is updated with
        `model_online` every `target_update` steps.
    last_update : int
        Tracks the number steps since the last update of `model_target`.
    use_DR : bool
        A flag indicating whether the Default Representation (DR) instead
        of the Successor Representation (SR) should be learned.
        By default set to false.
    use_follow_up_state : bool
        A flag indicating whether the SR/DR should be based on the
        follow-up state. By Default set to false.
    ignore_terminality : bool
        A flag indicating whether terminality should be ignored when
        learning the SR/DR. By Default set to true.
    gamma : float
        The agent's discount factor.
    M : DynaQMemory
        The memory module used by the Dyna-Q agent for storing
        environmental transitions.
        If none was provided in `memory` an instance with default
        parameters will be created.
    action_mask : NDArray
        A boolean 2D NumPy array of shape (`observation_space.n`, `action_space.n`)
        which represents an action mask that can be applied during action selection.
    mask_actions : bool
        A flag indicating whether action should be masked
        as defined in `action_mask`. Set to False by default.
    episodic_replay : bool
        A flag indicating whether replay should only occurr
        at the end of a trial. Set to False by default.

    Notes
    -----
    This agent only supports gym.spaces.Discrete for `observation_space`
    and `action_space`.

    Examples
    --------

    Here we initialize the Dyna-DSR agent for a discrete
    environment with 16 states and 4 actions. ::

        >>> import torch.nn as nn
        >>> import gymnasium as gym
        >>> from collections import OrderedDict
        >>> from cobel.agent import DynaDSR
        >>> from cobel.policy import EpsilonGreedy
        >>> from cobel.network import TorchNetwork
        >>> layers = [('dense', nn.Linear(16, 64)),
        ...           ('act', nn.ReLU()),
        ...           ('out', nn.Linear(64, 16))]
        >>> network = nn.Sequential(OrderedDict(layers))
        >>> agent = DynaDSR(gym.spaces.Discrete(16), gym.spaces.Discrete(4),
        ...         TorchNetwork(network), TorchNetwork(network), EpsilonGreedy(0.1))

    Training and testing of the agent is done with the train
    and test methods, respectively.
    Both methods expect an RL interface and number of trials
    as input.::

        >>> from cobel.interface import Gridworld
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> env = Gridworld(make_open_field(4, 4, 0, 1))
        >>> agent.train(env, 100)
        >>> agent.test(env, 100)

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        model_sr: Network,
        model_reward: Network,
        observations: None | NDArray = None,
        policy_test: None | Policy = None,
        gamma: float = 0.99,
        memory: None | DynaQMemory = None,
        custom_callbacks: None | CallbackDict = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Discrete, (
            'DynaDSR requires a discrete observation space!'
        )
        assert type(action_space) is gym.spaces.Discrete, (
            'DynaDSR requires a discrete action space!'
        )
        super().__init__(observation_space, action_space, custom_callbacks)
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        # build target and online models
        assert type(self.action_space) is gym.spaces.Discrete
        self.models_target = {a: model_sr.clone() for a in range(self.action_space.n)}
        self.models_online = {a: model_sr.clone() for a in range(self.action_space.n)}
        self.model_reward = model_reward
        # generate one-encoding if no observations were provided
        assert type(self.observation_space) is gym.spaces.Discrete
        self.observations: NDArray
        if observations is None:
            self.observations = np.eye(int(self.observation_space.n))
        else:
            self.observations = observations
        self.target_update: float = 10**-2
        self.last_update: int = 0
        self.use_DR: bool = False
        self.use_follow_up_state: bool = False
        self.ignore_terminality: bool = True
        self.gamma = gamma
        self.M: DynaQMemory
        if memory is None:
            self.M = DynaQMemory(int(observation_space.n), int(action_space.n))
        else:
            self.M = memory
        self.action_mask: NDArray = np.ones(
            (observation_space.n, action_space.n)
        ).astype(bool)
        self.mask_actions: bool = False
        self.episodic_replay: bool = False

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
            The batch size used for experience replay.
        no_replay : bool, default=False
            Flag indicating whether experience replay is disabled.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            state, _ = interface.reset()
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.policy.select_action(
                    self.retrieve_q(state),
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
                # update current state
                state = next_state
                # perform experience replay
                if not no_replay and not self.episodic_replay:
                    self.replay(batch_size)
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                # stop trial when the terminal state is reached
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            # perform experience replay
            if not no_replay and self.episodic_replay:
                self.replay(batch_size)
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
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
            The number of trials that the agent is tested.
        steps : int
            The maximum number of steps per trial.
        """
        for trial in range(trials):
            logs = self.callbacks.on_trial_begin(
                {
                    'trial_reward': 0.0,
                    'trial': self.current_trial,
                    'trial_session': trial,
                }
            )
            state, _ = interface.reset()
            assert type(state) is int
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                action = self.policy_test.select_action(
                    self.retrieve_q(state),
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
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def retrieve_q(self, state: int) -> NDArray:
        """
        This function retrieves the Q-values for a given state.

        Parameters
        ----------
        state : int
            The state for which Q-values will be retrieved.

        Returns
        -------
        q_values : NDArray
            The Q-values.
        """
        q_values = np.zeros(len(self.models_online))
        for action, model in self.models_online.items():
            sr = model.predict_on_batch(self.observations[state : (state + 1)])[0]  # type: ignore
            q_values[action] = self.model_reward.predict_on_batch(np.array([sr]))[0]  # type: ignore

        return q_values

    def predict_on_batch(self, batch: ArrayLike) -> NDArray:
        """
        This function retrieves the Q-values for a batch of observations.

        Parameters
        ----------
        batch : ArrayLike
            The batch of observations for which Q-values should be retrieved.

        Returns
        -------
        predictions : NDArray
            The batch of Q-value predictions.
        """
        q = np.zeros((np.array(batch).shape[0], len(self.models_online)))
        for action, model in self.models_online.items():
            sr = model.predict_on_batch(self.observations[np.array(batch)])
            q[:, action] = self.model_reward.predict_on_batch(sr).flatten()  # type: ignore

        return np.array(q)

    def replay(self, batch_size: int) -> None:
        """
        This function retrieves a batch of experiences and updates the Q-function.

        Parameters
        ----------
        batch_size : int
            The number of experiences that will be retrieved.
        """
        states: list[NDArray] = []
        actions: list[int] = []
        rewards: list[float] = []
        next_states: list[NDArray] = []
        terminals: list[bool] = []
        for experience in self.M.retrieve_batch(batch_size):
            states.append(self.observations[experience['state']])
            actions.append(experience['action'])
            rewards.append(experience['reward'])
            next_states.append(self.observations[experience['next_state']])
            terminals.append(bool(experience['terminal']))
        # compute the follow-up states' SR streams and values
        future_sr: dict[int, NDArray] = {}
        future_values: dict[int, NDArray] = {}
        for action, model in self.models_target.items():
            future_sr[action] = model.predict_on_batch(np.array(next_states))  # type: ignore
            future_values[action] = self.model_reward.predict_on_batch(
                future_sr[action]
            )  # type: ignore
        # compute targets
        inputs: dict[int, NDArray] = {}
        targets: dict[int, NDArray] = {}
        for action in self.models_target:
            # filter out irrelevant experiences
            idx = np.arange(len(actions)).astype(int)[np.array(actions) == action]
            inputs[action] = np.array(states)[idx]
            if self.use_follow_up_state:
                targets[action] = np.array(next_states)[idx]
            else:
                targets[action] = np.array(states)[idx]
            for i, index in enumerate(idx):
                # prepare bootstrap target
                bootstrap = (
                    next_states[index]
                    * (1 - self.use_follow_up_state)
                    * (1 - terminals[index])
                    * (1 - self.ignore_terminality)
                )
                # Deep SR
                if not self.use_DR:
                    best = np.argmax(
                        np.array([values[index] for _, values in future_values.items()])
                    )
                    bootstrap += future_sr[int(best)][index] * min(
                        terminals[index] + self.ignore_terminality, 1
                    )
                # Deep DR
                else:
                    bootstrap += np.mean(
                        np.array([SR[index] for _, SR in future_sr.items()]), axis=0
                    ) * min(terminals[index] + self.ignore_terminality, 1)
                targets[action][i] += self.gamma * bootstrap
        # update online models
        for action, _ in inputs.items():
            if inputs[action].shape[0] > 0:
                self.models_online[action].train_on_batch(
                    inputs[action], targets[action]
                )
        # update reward model
        self.model_reward.train_on_batch(np.array(next_states), np.array(rewards))
        # update target models
        self.last_update += 1
        if self.target_update < 1.0:
            for action in self.models_online:
                weights_target = np.array(
                    self.models_target[action].get_weights(), dtype=object
                )
                weights_online = np.array(
                    self.models_online[action].get_weights(), dtype=object
                )
                weights_target += self.target_update * (weights_online - weights_target)
                self.models_target[action].set_weights(weights_target)  # type: ignore
        elif self.last_update == self.target_update:
            for action in self.models_online:
                self.models_target[action].set_weights(
                    self.models_online[action].get_weights()
                )
            self.last_update = 0
