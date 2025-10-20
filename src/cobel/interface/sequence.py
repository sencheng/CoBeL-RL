# basic imports
import copy
import numpy as np
import gymnasium as gym
import pyqtgraph as pg  # type: ignore
# framework imports
from .interface import Interface
# typing
from .interface import Observation, StepTuple, ResetTuple, Action
from typing import TypedDict
from numpy.typing import NDArray


class TrialStep(TypedDict):
    observation: str
    reward: float | NDArray
    action: None | int


Trial = list[TrialStep]


class Sequence(Interface):
    """
    This class presents a predefined sequences of experiences to the agent.

    Parameters
    ----------
    trials : list of Trial
        A list containing the predefined trials of experiences.
        Each Trial consists of multiple TrialStep dictionaries which
        define observation, rewards and an optional action which
        can overwrite the agent's action.
    observations : dict of Observation
        A dictionary containing the observations referenced by the trial sequence.
    observation_space : gym.Space
        The observation space.
    nb_actions : int, default=1
        The number of actions that can be executed.
    overwrite : bool, default=False
        A flag indicating whether actions should be overwritten.
    widget : pg.GraphicsLayoutWidget or None, optional
        An optional widget. If provided the environment will be visualized.

    Attributes
    ----------
    trials : list of Trial
        A list containing the predefined trials of experiences.
        Each Trial consists of multiple TrialStep dictionaries which
        define observation, rewards and an optional action which
        can overwrite the agent's action.
    observations : dict of Observation
        A dictionary containing the observations referenced by the trial sequence.
    observation_space : gym.Space
        The observation space.
    nb_actions : int, default=1
        The number of actions that can be executed.
    overwrite : bool, default=False
        A flag indicating whether actions should be overwritten.
    current_trial : int
        Tracks the current trial.
    current_step : int
        Tracks the current trial step.
    current_observation : Observation
        The current observation.

    Examples
    --------

    Trials sequences and observations can be easily predefined
    and used to train RL agents. ::

        >>> from cobel.interface import Sequence
        >>> observations = {'s_1': np.eye(2)[0],
        ...                 's_2': np.eye(2)[1]}
        >>> trials = [[{'observation': 's_1',
        ...             'reward': np.array([0., 1.]),
        ...             'action:' None}],
        ...           [{'observation': 's_1',
        ...             'reward': np.array([1., 0.]),
        ...             'action': None}]]
        >>> env = Sequence(trials * 2, observations,
        ...                gym.spaces.Box(0, 1, (2,)))

    """

    def __init__(
        self,
        trials: list[Trial],
        observations: dict[str, Observation],
        observation_space: gym.spaces.Space,
        nb_actions: int = 1,
        overwrite: bool = False,
        widget: None | pg.GraphicsLayoutWidget = None,
    ) -> None:
        super().__init__(widget)
        # the trial sequence
        self.trials = trials
        self.overwrite = overwrite
        # the observations
        self.observations = observations
        self.current_observation: Observation
        # the current trial
        self.current_trial = 0
        # the current trial step
        self.current_step = 0
        # prepare observation and action spaces
        self.observation_space = observation_space
        self.action_space = gym.spaces.Discrete(nb_actions)
        self.zero_current()

    def zero_current(self) -> None:
        """
        This function zeros the current observation.
        """
        if type(self.observation_space) is gym.spaces.Dict:
            obs = list(self.observations.values())[0]
            assert type(obs) is dict
            self.current_observation = {i: np.zeros(o.shape) for i, o in obs.items()}
        elif type(self.observation_space) is gym.spaces.Box:
            assert type(self.observations) is dict
            self.current_observation = np.zeros(
                list(self.observations.values())[0].shape  # type: ignore
            )
        elif type(self.observation_space) is gym.spaces.Discrete:
            self.current_observation = 0

    def step(self, action: Action) -> StepTuple:
        """
        The interface's step function (compatible with Gymnasium's step function).

        Parameters
        ----------
        action : Action
            The action selected by the agent.

        Returns
        -------
        observation : Observation
            The observation of the new current state.
        reward : float
            The reward received.
        end_trial : bool
            A flag indicating whether the trial ended.
        truncated : bool
            A flag required by Gymnasium (not used).
        logs : dict
            The (empty) logs dictionary.
        """
        # retrieve reward/reinforcement
        self.zero_current()
        reward: float
        a: int = int(action)
        step_reward = self.trials[self.current_trial][self.current_step]['reward']
        step_action = self.trials[self.current_trial][self.current_step]['action']
        if type(step_reward) is float:
            reward = step_reward
        else:
            if self.overwrite:
                assert step_action is not None
                a = step_action
            reward = step_reward[a]  # type: ignore
        # update trial step
        self.current_step += 1
        # determine this was the last trial step
        end_trial = len(self.trials[self.current_trial]) == self.current_step
        # retrieve observation only if a next step exists
        if not end_trial:
            self.current_observation = copy.deepcopy(
                self.observations[
                    self.trials[self.current_trial][self.current_step]['observation']
                ]
            )
        # increment current trial here to prevent redundant reset calls from doing so
        self.current_trial += end_trial

        return (
            copy.deepcopy(self.current_observation),
            reward,
            end_trial,
            end_trial,
            {'action': a, 'step_action': step_action},
        )

    def reset(self) -> ResetTuple:
        """
        The interface's reset function (compatible with Gymnasium's reset function).

        Returns
        -------
        observation : Observation
            The observation of the new current state.
        logs : dict
            The (empty) logs dictionary.
        """
        self.current_observation = copy.deepcopy(
            self.observations[self.trials[self.current_trial][0]['observation']]
        )
        self.current_step = 0

        return copy.deepcopy(self.current_observation), {}

    def get_position(self) -> NDArray:
        """
        This function returns the agent's position in the environment.

        Returns
        -------
        position : NDArray
            A numpy array containing the agent's position.
        """
        return np.array([])
