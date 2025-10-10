# basic imports
import numpy as np
import gymnasium as gym
# framework imports
from .agent import Agent
from ..policy.policy import Policy
from ..interface.interface import Interface
# typing
from .agent import CallbackDict
from numpy.typing import NDArray, ArrayLike


class AssociativeNetwork(Agent):
    """
    This class implements the associative net model from Donoso et al. (2021):
    https://doi.org/10.1007/s10071-021-01521-4

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
    saturation : float or dict of NDArray, default=0.9
        The saturation level of the weights.
    learning_rate : float or dict of NDArray, default=0.01
        The learning rates of the weights.
    noise : float, default=1.
        The amplitude of the noise that is added to the output units.
    linear_update : bool, default=False
        Flag indicating whether linear update should be used.
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
    weights : dict of NDArray
        A dictionary containing the excitatory and
        inhibitory weights of the associative network.
    saturation : dict of NDArray
        The saturation level of the weights.
    learning_rate : float or dict of NDArray
        The learning rates of the weights.
    linear_update : bool
        Flag indicating whether linear update should be used.
        Linear updates ignore the weight saturation.
    noise_amplitude : float
        The amplitude of the noise that is added to the output units.
    alpha : float
        Related to associativity (?). Part of the original Matlab code
        but was unusued and not mentioned in the Donoso et al. (2021).
    d_alpha : float
        Related to associativity (?). Part of the original Matlab code
        but was unusued and not mentioned in the Donoso et al. (2021).
    current_trial : int
        Tracks the current trial.
    rng : numpy.random.Generator
        A random number generator instance used for
        probablistic action selection.

    Notes
    -----
    This agent only supports gym.spaces.Box for `observation_space`
    and gym.spaces.Discrete for `action_space`.

    Examples
    --------

    Here we initialize the agent for a Sequence environment
    which presents observations of shape (10, ) and has 2 actions. ::

        >>> import gymnasium as gym
        >>> from cobel.agent import AssociativeNetwork
        >>> from cobel.policy import EpsilonGreedy
        >>> agent = AssociativeNetwork(gym.spaces.Box(0., 1., (10, )),
        ...         gym.spaces.Discrete(2), EpsilonGreedy(0.1))

    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Policy,
        policy_test: None | Policy = None,
        saturation: float | dict[str, NDArray] = 20.0,
        learning_rate: float | dict[str, NDArray] = 0.01,
        noise: float = 1.0,
        linear_update: bool = False,
        custom_callbacks: None | CallbackDict = None,
        rng: None | np.random.Generator = None,
    ) -> None:
        assert type(observation_space) is gym.spaces.Box, 'Wrong observation space!'
        assert type(action_space) is gym.spaces.Discrete, 'Wrong action space!'
        super().__init__(observation_space, action_space, custom_callbacks)
        assert type(self.observation_space.shape) is tuple
        assert type(self.action_space) is gym.spaces.Discrete
        self.policy = policy
        self.policy_test = policy if policy_test is None else policy_test
        self.rng = np.random.default_rng() if rng is None else rng
        # initialize weights
        self.weights = {
            'excitatory': np.zeros(
                (np.prod(self.observation_space.shape), self.action_space.n - 1)
            ),
            'inhibitory': np.zeros(
                (np.prod(self.observation_space.shape), self.action_space.n - 1)
            ),
        }
        # define saturation values for all weights
        self.saturation: dict[str, NDArray]
        if type(saturation) is dict:
            self.saturation = saturation
        else:
            self.saturation = {
                'excitatory': np.full(
                    (np.prod(self.observation_space.shape), self.action_space.n - 1),
                    saturation,
                ),
                'inhibitory': np.full(
                    (np.prod(self.observation_space.shape), self.action_space.n - 1),
                    saturation,
                ),
            }
        # define learning rates for all weights
        self.learning_rate: dict[str, NDArray]
        if type(learning_rate) is dict:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = {
                'excitatory': np.full(
                    (np.prod(self.observation_space.shape), self.action_space.n - 1),
                    learning_rate,
                ),
                'inhibitory': np.full(
                    (np.prod(self.observation_space.shape), self.action_space.n - 1),
                    learning_rate,
                ),
            }
        # if true, saturation is ignored when updating the weights
        self.linear_update = linear_update
        # noise strength
        self.noise_amplitude = noise
        # scales the weight update? wasn't mentioned in the paper
        self.alpha = 1.0
        self.d_alpha = 0.0
        # session trial
        self.current_trial = 0

    def train(
        self, interface: gym.Env | Interface, trials: int, steps: int = 32
    ) -> None:
        """
        This function is called to train the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is trained.
        steps : int, default=32
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
            s, _ = interface.reset()
            assert type(s) is np.ndarray, 'Invalid observation type'
            state = s.flatten()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                q = self.retrieve_q(state)
                action = self.policy.select_action(q)
                ns, reward, end_trial, truncated, log = interface.step(action)
                assert type(ns) is np.ndarray, 'Invalid observation type'
                next_state = ns.flatten()
                # update Q-function amd stpre experience
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                self.update_q(experience)
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def test(
        self, interface: gym.Env | Interface, trials: int, steps: int = 32
    ) -> None:
        """
        This function is called to test the agent.

        Parameters
        ----------
        interface : gym.Env or Interface
            The environment that the agent interacts with.
        trials : int
            The number of trials that the agent is tested.
        steps : int, default=32
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
            s, _ = interface.reset()
            assert type(s) is np.ndarray, 'Invalid observation type'
            state = s.flatten()
            for step in range(steps):
                # update logs
                logs['step'] = step
                logs = self.callbacks.on_step_begin(logs)
                # select action and execute it
                q = self.retrieve_q(state)
                action = self.policy.select_action(q)
                ns, reward, end_trial, truncated, log = interface.step(action)
                assert type(ns) is np.ndarray, 'Invalid observation type'
                next_state = ns.flatten()
                # update Q-function amd stpre experience
                experience = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'terminal': (1 - end_trial),
                }
                state = next_state
                # update logs
                logs['trial_reward'] += reward
                logs.update(experience)
                logs = self.callbacks.on_step_end(logs)
                if end_trial:
                    break
            self.current_trial += 1
            logs['steps'] = step
            logs = self.callbacks.on_trial_end(logs)
            # force stop the training session
            if self.stop:
                break

    def rescale_weights(self, factor: dict[str, float]) -> None:
        """
        This function rescales the excitatory and inhibitory weights of the network.

        Parameters
        ----------
        factor : dict of float
            A dictionary containing the rescaling factors
            for excitatory and inhibitory weights.
        """
        self.weights['excitatory'] *= factor['excitatory']
        self.weights['inhibitory'] *= factor['inhibitory']

    def retrieve_q(self, observation: NDArray) -> NDArray:
        """
        This function predicts Q-values for a given observation.

        Parameters
        ----------
        observation : NDArray
            The observation for which Q-values should be retrieved.

        Returns
        -------
        q_values : NDArray
            The predicated Q-values.
        """
        noise = self.noise_amplitude * self.rng.random(
            self.weights['excitatory'].shape[1]
        )

        return (
            np.matmul(observation, self.weights['excitatory'])
            - np.matmul(observation, self.weights['inhibitory'])
            + noise
        )

    def update_q(self, experience: dict) -> None:
        """
        This function updates the agent with a given experience.

        Parameters
        ----------
        experience : dict
            The experience with which the network will be updated.
        """
        # compute weight update mask
        action_vector = (
            np.arange(self.weights['excitatory'].shape[1]).astype(int)
            == experience['action']
        ).astype(int)
        update_mask = np.outer((experience['state'] != 0).astype(int), action_vector)
        # strengthen excitatory weights for correct response,
        # inhibitory weights for incorrect response
        weight = 'excitatory' if experience['reward'] > 0 else 'inhibitory'
        delta = self.alpha * (self.saturation[weight] - self.weights[weight])
        if self.linear_update:
            delta.fill(1.0)
        self.weights[weight] += self.learning_rate[weight] * delta * update_mask

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
        return np.array(
            [self.retrieve_q(observation) for observation in np.array(batch)]
        )
