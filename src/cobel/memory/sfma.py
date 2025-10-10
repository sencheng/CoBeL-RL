# basic imports
import numpy as np
# framework imports
from .utils.metrics import Metric
# typing
from typing import TypedDict, NotRequired
from numpy.typing import NDArray


class Experience(TypedDict):
    state: int
    action: int
    reward: float
    next_state: int
    terminal: int
    td: NotRequired[float]


class SFMAMemory:
    """
    Memory module to be used with the SFMA agent.
    Experiences are stored as a static table.

    Parameters
    ----------
    metric : Metric
        The metric used to compute the similarity between experiences.
    nb_states : int
        The number of environmental states.
    nb_actions : int
        The number of the agent's actions.
    decay_inhibition : float, default=0.9
        The factor by which inhibition is decayed.
    decay_strength : float, default=1.
        The factor by which the experience strengths are decayed.
    learning_rate : float, default=0.9
        The learning rate with which experiences are updated.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    nb_states : int
        The number of environmental states.
    nb_actions : int
        The number of the agent's actions.
    decay_inhibition : float
        The factor by which inhibition is decayed.
    decay_strength : float
        The factor by which the experience strengths are decayed.
    decay_recency : float
        The factor by which the experience recencies are decayed.
    learning_rate : float
        The learning rate with which experiences are updated.
    beta : float
        The inverse temperature used for the softmax.
    reward_mod_local : bool
        A flag indicating whether reward locally modulates experience strength.
    error_mod_local : bool
        A flag indicating whether TD-error locally modulates experience strength.
    reward_mod : bool
        A flag indicating whether reward modulates experience strength.
    error_mod : bool
        A flag indicating whether TD-error modulates experience strength.
    policy_mod : bool
        A flag indicating whether the policy modulates experience strength.
    state_mod : bool
        A flag indicating whether the state modulates experience strength.
    metric : Metric
        The metric used to compute the similarity between experiences.
    rewards : NDArray
        Stores the rewards for each environmental transition.
    states : NDArray
        Stores the follow-up state for each environmental transition.
    terminals : NDArray
        Stores the terminality for each environmental transition.
    C : NDArray
        Stores the experience strengths.
    T : NDArray
        Stores the experience recencies.
    I : NDArray
        Stores the experience inhibition.
    C_step : float
        The amount by which experience strengths are increased.
    I_step : float
        The amount by which experience inhibition is increased.
    R_threshold : float
        The priority rating threshold.
    deterministic : bool
        A flag indicating whether experiences are reactivated
        in a deterministic fashion.
    recency : bool
        A flag indicating whether recency is included in
        the calculation of priority ratings.
    C_normalize : bool
        A flag indicating whether experience strengths are normalized
        before recency calculation.
    D_normalize : bool
        A flag indicating whether experience similarities are normalized
        before recency calculation.
    R_normalize : bool
        A flag indicating whether priority ratings are normalized.
    mode : str
        The replay mode that is used. Can be 'default', 'reverse', 'forward',
        'blend_reverse', 'blend_forward', 'interpolate' or 'sweeping'.
        Defaults to 'default'.
    reward_modulation : float
        The amount by which rewards are scaled for reward modulation.
    blend : float
        The amount of blending when `mode` is 'blender_reverse' or 'blend_forward'.
    interpolation_fwd : float
        The amount of forward mode interpolation when `mode` is 'interpolate'.
    interpolation_rev : float
        The amount of reverse mode interpolation when `mode` is 'interpolate'.
    rng : numpy.random.Generator
        A random number generator instance used for probablistic replay.

    Examples
    --------

    Initializing the memory module for a simple gridworld. ::

        >>> from cobel.memory import SFMAMemory
        >>> from cobel.memory.utils.metrics import SR
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> world = make_open_field(4, 4)
        >>> metric = SR(wordl['sas'], 0.9)
        >>> memory = SFMAMemory(metric, 16, 4)

    """

    def __init__(
        self,
        metric: Metric,
        nb_states: int,
        nb_actions: int,
        decay_inhibition: float = 0.9,
        decay_strength: float = 1.0,
        learning_rate: float = 0.9,
        rng: None | np.random.Generator = None,
    ) -> None:
        self.rng = np.random.default_rng() if rng is None else rng
        # initialize variables
        self.nb_states = nb_states
        self.nb_actions = nb_actions
        self.decay_inhibition = decay_inhibition
        self.decay_strength = decay_strength
        self.decay_recency = 0.9
        self.learning_rate = learning_rate
        self.beta = 20
        self.rlAgent = None
        # experience strength modulation parameters
        self.reward_mod_local = False  # increase during experience
        self.error_mod_local = False  # increase during experience
        self.reward_mod = False  # increase during experience
        self.error_mod = False  # increase during experience
        self.policy_mod = False  # added before replay
        self.state_mod = False
        # similarity metric
        self.metric = metric
        # prepare memory structures
        self.rewards = np.zeros((self.nb_states, self.nb_actions))
        self.states = np.tile(
            np.arange(self.nb_states).reshape(self.nb_states, 1), self.nb_actions
        ).astype(int)
        self.terminals = np.zeros((self.nb_states, self.nb_actions)).astype(int)
        # prepare replay-relevant structures
        self.C = np.zeros(self.nb_states * self.nb_actions)  # strength
        self.T = np.zeros(self.nb_states * self.nb_actions)  # recency
        self.I = np.zeros(self.nb_states)  # inhibition
        # increase step size
        self.C_step = 1.0
        self.I_step = 1.0
        # priority rating threshold
        self.R_threshold = 10.0**-6
        # always reactive experience with highest priority rating
        self.deterministic = False
        # consider recency of experience
        self.recency = False
        # normalize variables
        self.C_normalize = False
        self.D_normalize = False
        self.R_normalize = True
        # replay mode
        self.mode = 'default'
        # modulates reward
        self.reward_modulation = 1.0
        # weighting of forward/reverse mode when using blending modes
        self.blend = 0.1
        # weightings of forward abdreverse modes when using interpolation mode
        self.interpolation_fwd, self.interpolation_rev = 0.5, 0.5

    def store(self, experience: Experience) -> None:
        """
        This function stores a given experience.

        Parameters
        ----------
        experience : dict
            The experience to be stored.
        """
        state, action = experience['state'], experience['action']
        # update experience
        self.rewards[state][action] += self.learning_rate * (
            experience['reward'] - self.rewards[state][action]
        )
        self.states[state][action] = experience['next_state']
        self.terminals[state][action] = experience['terminal']
        # update replay-relevent structures
        self.C *= self.decay_strength
        self.C[self.nb_states * action + state] += self.C_step
        self.T *= self.decay_recency
        self.T[self.nb_states * action + state] = 1.0
        # local reward modulation (affects this experience only)
        if self.reward_mod_local:
            self.C[self.nb_states * action + state] += (
                experience['reward'] * self.reward_modulation
            )
        # reward modulation (affects close experiences)
        if self.reward_mod:
            modulation = np.tile(self.metric.D[experience['state']], self.nb_actions)
            self.C += experience['reward'] * modulation * self.reward_modulation
        # local RPE modulation (affects this experience only)
        if self.error_mod_local:
            self.C[self.nb_states * action + state] += np.abs(experience['td'])
        # RPE modulation (affects close experiences)
        if self.error_mod:
            modulation = np.tile(
                self.metric.D[experience['next_state']], self.nb_actions
            )
            self.C += np.abs(experience['td']) * modulation
        # additional strength increase of all experiences at current state
        if self.state_mod:
            self.C[[state + self.nb_states * a for a in range(self.nb_actions)]] += 1.0

    def replay(
        self,
        replay_length: int,
        current_state: None | int = None,
        current_action: None | int = None,
    ) -> list[Experience]:
        """
        This function replays experiences using SFMA.

        Parameters
        ----------
        replay_length : int
            The number of experiences that will be replayed.
        current_state : int or None, optional
            The state at which replay should start.
        current_action : int or None, optionald
            The action with which replay should start.

        Returns
        -------
        experiences : list of dict
            The replay batch.
        """
        action = current_action
        # if no action is specified pick one at random
        if current_action is None:
            action = int(self.rng.integers(self.nb_actions))
        # if a state is not defined, then choose an experience according
        # to relative experience strengths
        if current_state is None:
            # we clip the strengths to catch negative values caused by rounding errors
            P = np.clip(self.C, a_min=0, a_max=None) / np.sum(
                np.clip(self.C, a_min=0, a_max=None)
            )
            exp = self.rng.choice(np.arange(0, P.shape[0]), p=P)
            current_state = exp % self.nb_states
            action = int(exp / self.nb_states)
        next_state = self.states[current_state, action]
        # reset inhibition
        self.I *= 0
        # replay
        experiences = []
        for _ in range(replay_length):
            # retrieve experience strengths
            C = np.copy(self.C)
            if self.C_normalize:
                C /= np.amax(C)
            # retrieve experience similarities
            D = np.tile(self.metric.D[current_state], self.nb_actions)
            if self.D_normalize:
                D /= np.amax(D)
            if self.mode == 'forward':
                D = np.tile(self.metric.D[next_state], self.nb_actions)
            elif self.mode == 'reverse':
                D = D[self.states.flatten(order='F')]
            elif self.mode == 'blend_forward':
                D += self.blend * np.tile(self.metric.D[next_state], self.nb_actions)
            elif self.mode == 'blend_reverse':
                D += self.blend * D[self.states.flatten(order='F')]
            elif self.mode == 'interpolate':
                D = (
                    self.interpolation_fwd
                    * np.tile(self.metric.D[next_state], self.nb_actions)
                    + self.interpolation_rev * D[self.states.flatten(order='F')]
                )
            elif self.mode == 'sweeping':
                D = np.tile(self.metric.D[next_state], self.nb_actions)[
                    self.states.flatten(order='F')
                ]
            # retrieve inhibition
            I = np.tile(self.I, self.nb_actions)
            # compute priority ratings
            R = C * D * (1 - I)
            if self.recency:
                R *= self.T
            # apply threshold to priority ratings
            R[R < self.R_threshold] = 0.0
            # stop replay sequence if all priority ratings are all zero
            if np.sum(R) == 0.0:
                break
            # determine state and action
            if self.R_normalize:
                R /= np.amax(R)
            exp = np.argmax(R)
            if not self.deterministic:
                # compute activation probabilities
                probs = self.softmax(R, -1, self.beta)
                probs = probs / np.sum(probs)
                exp = self.rng.choice(np.arange(0, probs.shape[0]), p=probs)
            # determine experience tuple
            action = int(exp / self.nb_states)
            current_state = exp - (action * self.nb_states)
            next_state = self.states[current_state][action]
            # apply inhibition
            self.I *= self.decay_inhibition
            self.I[current_state] = min(float(self.I[current_state] + self.I_step), 1.0)
            # "reactivate" experience
            experience: Experience = {
                'state': current_state,
                'action': action,
                'reward': self.rewards[current_state][action],
                'next_state': next_state,
                'terminal': self.terminals[current_state][action],
            }
            experiences += [experience]
            # stop replay at terminal states
            # if experience['terminal']:
            #    break

        return experiences

    def softmax(self, data: NDArray, offset: float = 0, beta: float = 5) -> NDArray:
        """
        This function computes the custom softmax over the input.

        Parameters
        ----------
        data : NDArray
            Input of the softmax function.
        offset : float, default=0
            Offset added after applying the softmax function.
        beta : float, default=5
            Beta value.

        Returns
        -------
        priorities : NDArray
            The softmax priorities.
        """
        exp = np.exp(data * beta) + offset
        if np.sum(exp) == 0:
            exp.fill(1)
        else:
            exp /= np.sum(exp)

        return exp

    def retrieve_random_batch(
        self, number_of_experiences: int, mask: NDArray
    ) -> list[Experience]:
        """
        This function retrieves a number of random experiences.

        Parameters
        ----------
        number_of_experiences : int
            The number of random experiences to be drawn.
        mask : NDArray
            Masks invalid transitions.

        Returns
        -------
        experiences : list of dict
            The replay batch.
        """
        # draw random experiences
        probs = np.ones(self.nb_states * self.nb_actions) * mask.astype(int)
        probs /= np.sum(probs)
        idx = self.rng.choice(
            np.arange(self.nb_states * self.nb_actions), number_of_experiences, p=probs
        )
        # determine indeces
        transitions = np.unravel_index(
            idx, (self.nb_states, self.nb_actions), order='F'
        )
        # build experience batch
        experiences: list[Experience] = []
        for state, action in transitions:
            experiences += [
                {
                    'state': state,
                    'action': action,
                    'reward': self.rewards[state][action],
                    'next_state': self.states[state][action],
                    'terminal': self.terminals[state][action],
                }
            ]

        return experiences
