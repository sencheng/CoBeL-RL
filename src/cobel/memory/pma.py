# basic imports
import numpy as np
from scipy import linalg  # type: ignore
# typing
from typing import TypedDict, NotRequired
from numpy.typing import NDArray
from ..policy.policy import Policy


class Experience(TypedDict):
    state: int
    action: int
    reward: float
    next_state: int
    terminal: int
    td: NotRequired[float]


class PMAMemory:
    """
    Memory module to be used with the PMA agent.
    Experiences are stored as a static table.

    Parameters
    ----------
    sas : NDArray
        The state-action-state transition matrix (uniform action policy).
    policy : Policy
        The action selection policy used by the agent.
    learning_rate : float, default=0.9
        The learning rate with which experiences are updated.
    learning_rate_q : float, default=0.9
        The learning rate used by the agent to update its Q-function.
    gamma : float, default=0.9
        The discount factor that is used to compute the successor representation.
    gamma_q : float, default=0.9
        The discount factor used by the agent to update its Q-function.
    rng : numpy.random.Generator or None, optional
        An optional random number generator instance.
        If none is provided a new instance will be created.

    Attributes
    ----------
    sas : NDArray
        The state-action-state transition matrix (uniform action policy).
    policy : Policy
        The action selection policy used by the agent.
    learning_rate : float
        The learning rate with which experiences are updated.
    learning_rate_q : float
        The learning rate used by the agent to update its Q-function.
    gamma : float
        The discount factor that is used to compute the successor representation.
    gamma_q : float
        The discount factor used by the agent to update its Q-function.
    nb_states : int
        The number of environmental states.
    nb_action : int
        The number of agent's actions.
    min_gain : float
        The minimum update gain value.
    min_gain_mode : str
        The minimum gain calculation mode. When set to 'original' (default)
        minimum gain values are summed for each experience in a sequence.
        Otherwise minimum gain is applied once for the whole sequence.
    equal_need : bool
        A flag indicating whether the need term should be set uniform.
        Defaults to False.
    equal_gain : bool
        A flag indicating whether the gain term should be set uniform.
        Defaults to False.
    ignore_barriers : bool
        A flag indicating whether transitions leading into barriers
        should be ignored during replay. Defaults to True.
    allow_loops : bool
        A flag indicating whether replay allows looping/intersecting
        sequences. Defaults to False.
    rewards : NDArray
        Stores the rewards for each environmental transition.
    states : NDArray
        Stores the follow-up state for each environmental transition.
    terminals : NDArray
        Stores the terminality for each environmental transition.
    T : NDArray
        The state-state transition matrix.
    SR : NDArray
        The successor representation matrix.
    update_mask : NDArray
        Masks transitions that should be ignored during replay.
    rng : numpy.random.Generator
        A random number generator instance used for probablistic replay.

    Examples
    --------

    Initializing the memory module for a simple gridworld. ::

        >>> from cobel.memory import PMAMemory
        >>> from cobel.policy import EpsilonGreedy
        >>> from cobel.misc.gridworld_tools import make_open_field
        >>> world = make_open_field(4, 4)
        >>> policy = EpsilonGreedy(0.1)
        >>> memory = PMAMemory(world['sas'], policy)

    """

    def __init__(
        self,
        sas: NDArray,
        policy: Policy,
        learning_rate: float = 0.9,
        learning_rate_q: float = 0.9,
        gamma: float = 0.9,
        gamma_q: float = 0.9,
        rng: None | np.random.Generator = None,
    ) -> None:
        self.rng = np.random.default_rng() if rng is None else rng
        # initialize variables
        self.sas = sas
        self.policy = policy
        self.learning_rate = learning_rate
        self.learning_rate_q = learning_rate_q
        self.learning_rate_T = 0.9
        self.gamma = gamma
        self.gamma_q = gamma_q
        self.nb_states = sas.shape[0]
        self.nb_actions = sas.shape[1]
        self.min_gain = 10**-6
        self.min_gain_mode = 'original'
        self.equal_need = False
        self.equal_gain = False
        self.ignore_barriers = True
        self.allow_loops = False
        # initialize memory structures
        self.rewards = np.zeros((self.nb_states, self.nb_actions))
        self.states = np.zeros((self.nb_states, self.nb_actions)).astype(int)
        self.terminals = np.zeros((self.nb_states, self.nb_actions)).astype(int)
        # compute state-state transition matrix
        self.T = np.sum(self.sas, axis=1) / self.nb_actions
        # compute successor representation
        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)
        # determine transitions that should be ignored
        # (i.e. those that lead into the same state)
        self.update_mask = self.states.flatten(order='F') != np.tile(
            np.arange(self.nb_states), self.nb_actions
        )

    def store(self, experience: Experience) -> None:
        """
        This function stores a given experience.

        Parameters
        ----------
        experience : Experience
            The experience to be stored.
        """
        # update experience
        s, a, r = experience['state'], experience['action'], experience['reward']
        self.rewards[s][a] += self.learning_rate * (r - self.rewards[s][a])
        self.states[s][a] = experience['next_state']
        self.terminals[s][a] = experience['terminal']
        # update T
        self.T[experience['state']] += self.learning_rate_T * (
            (np.arange(self.nb_states) == experience['next_state'])
            - self.T[experience['state']]
        )

    def replay(
        self,
        q_function: NDArray,
        action_mask: None | NDArray,
        replay_length: int,
        current_state: None | int,
        force_first: None | int = None,
    ) -> tuple[list, NDArray]:
        """
        This function replays experiences.

        Parameters
        ----------
        q_function : NDArray
            The Q-function of the agent.
        action_mask : NDArray or None
            The action mask used by the agent.
        replay_length : int
            The number of experiences that will be replayed.
        current_state : int or None
            State at which replay should start.
        force_first : int or None, optional
            If a state is specified, replay is forced to start here.

        Returns
        -------
        updates : list of Experience
            The updates that were executed.
        q_function : NDArray
            The updated Q-function.
        """
        performed_updates: list[Experience] = []
        Q = np.copy(q_function)  # noqa: N806
        last_seq = 0
        for update in range(replay_length):
            # make list of 1-step backups
            updates: list[list[Experience]] = []
            for i in range(self.nb_states * self.nb_actions):
                s, a = i % self.nb_states, int(i / self.nb_states)
                updates += [
                    [
                        {
                            'state': s,
                            'action': a,
                            'reward': self.rewards[s, a],
                            'next_state': self.states[s, a],
                            'terminal': self.terminals[s, a],
                        }
                    ]
                ]
            # extend current update sequence
            extend = -1
            if len(performed_updates) > 0:
                # determine extending state
                extend = performed_updates[-1]['next_state']
                # check for loop
                loop = False
                for step in performed_updates[last_seq:]:
                    if extend == step['state']:
                        loop = True
                # extend update
                if not loop or self.allow_loops:
                    # determine extending action which yields max future value
                    mask = action_mask[extend] if action_mask is not None else None
                    extending_action = self.policy.select_action(Q[extend], mask)
                    # determine extending step
                    extend += int(extending_action) * self.nb_states
                    updates[extend] = performed_updates[last_seq:] + updates[extend]
            # compute gain and need
            gain = self.compute_gain_batch(Q, action_mask)  # ! check for consistency
            if extend != -1:
                gain[extend] = self.compute_gain(Q, action_mask, [updates[extend]])
            # gain = self.compute_gain(updates) # (old) iterative version
            if self.equal_gain:
                gain.fill(1)
            need = self.compute_need(current_state)
            if self.equal_need:
                need.fill(1)
            # determine backup with highest utility
            utility = gain * need
            if self.ignore_barriers:
                utility *= self.update_mask
            # determine backup with highest utility
            ties = utility == np.amax(utility)
            utility_max = self.rng.choice(
                np.arange(self.nb_states * self.nb_actions), p=ties / np.sum(ties)
            )
            # force initial update
            if len(performed_updates) == 0 and force_first is not None:
                utility_max = (
                    force_first + self.rng.integers(self.nb_actions) * self.nb_states
                )
            # perform update
            Q = self.update_q(Q, updates[utility_max])  # noqa: N806
            # add update to list
            performed_updates += [updates[utility_max][-1]]
            if extend != utility_max:
                last_seq = update

        return performed_updates, Q

    def compute_gain(
        self,
        q_function: NDArray,
        action_mask: None | NDArray,
        updates: list[list[Experience]],
    ) -> NDArray:
        """
        This function computes the gain for each possible n-step backup in updates.

        Parameters
        ----------
        Q : NDArray
            The Q-function of the agent.
        action_mask : NDArray or None
            The action mask used by the agent.
        updates : list of Experience
            A list of n-step updates.

        Returns
        -------
        gains : NDArray
            The gain values for the n-step updates.
        """
        gains = []
        for update in updates:
            gain = 0.0
            # expected future value
            future_value = (
                np.amax(q_function[update[-1]['next_state']]) * update[-1]['terminal']
            )
            # gain is accumulated over the whole trajectory
            for s, step in enumerate(update):
                # policy before update
                mask = action_mask[step['state']] if action_mask is not None else None
                policy_before = self.policy.get_action_probs(
                    q_function[step['state']], mask
                )
                # sum rewards over subsequent n-steps
                r = 0.0
                for following_steps in range(len(update) - s):
                    r += update[s + following_steps]['reward'] * (
                        self.gamma**following_steps
                    )
                # compute new Q-value
                q_target = np.copy(q_function[step['state']])
                q_target[step['action']] = r + future_value * (
                    self.gamma_q ** (following_steps + 1)
                )
                q_new = q_function[step['state']] + self.learning_rate_q * (
                    q_target - q_function[step['state']]
                )
                # policy after update
                policy_after = self.policy.get_action_probs(q_new, mask)
                # compute gain
                step_gain = np.sum(q_new * policy_after) - np.sum(q_new * policy_before)
                if self.min_gain_mode == 'original':
                    step_gain = max(step_gain, self.min_gain)
                # add gain
                gain += step_gain
            # store gain for current update
            gains += [max(gain, self.min_gain)]

        return np.array(gains)

    def compute_gain_batch(
        self, q_function: NDArray, action_mask: None | NDArray
    ) -> NDArray:
        """
        This function computes the gain for each possible 1-step backup.

        Parameters
        ----------
        Q : NDArray
            The Q-function of the agent.
        action_mask : NDArray or None
            The action mask used by the agent.

        Returns
        -------
        gains : NDArray
            The gain values for the 1-step updates.
        """
        # prepare updates
        updates = np.tile(q_function, (q_function.shape[1], 1))
        # bootstrap target values
        next_states = self.states.flatten(order='F')
        targets = q_function[next_states]
        # prepare target mask
        target_mask = np.zeros(updates.shape)
        for action in range(q_function.shape[1]):
            start = q_function.shape[0] * action
            stop = q_function.shape[0] * (action + 1)
            target_mask[start:stop, action] = 1.0
        # compute updated Q-values
        q_new = np.copy(updates)
        q_new += (
            self.learning_rate_q
            * target_mask
            * (
                np.tile(self.rewards, (q_function.shape[1], 1))
                + self.gamma_q
                * np.amax(targets, axis=1).reshape(targets.shape[0], 1)
                * self.terminals.flatten(order='F').reshape(targets.shape[0], 1)
                - q_new
            )
        )
        # compute policies pre and post update
        mask = (
            np.tile(action_mask, (q_function.shape[1], 1))
            if action_mask is not None
            else None
        )
        policy_old = self.action_probs_batch(updates, mask)
        policy_new = self.action_probs_batch(q_new, mask)
        # comute gain for all updates
        gain = np.sum(policy_new * q_new, axis=1) - np.sum(policy_old * q_new, axis=1)

        return np.clip(gain, a_min=self.min_gain, a_max=None)

    def compute_need(self, current_state: None | int = None) -> NDArray:
        """
        This function computes the need for each possible n-step backup in updates.

        Parameters
        ----------
        current_state : int or None, optional
            The state that the agent currently is in.

        Returns
        -------
        needs : NDArray
            The gain values for the n-step updates.
        """
        # use standing distribution of the MDP for 'offline' replay
        if current_state is None:
            # compute left eigenvectors
            eig, vec = linalg.eig(self.T, left=True, right=False)
            best = np.argmin(np.abs(eig - 1))

            return np.tile(np.abs(vec[:, best].T), self.nb_actions)
        # use SR given the current state for 'awake' replay
        else:
            return np.tile(self.SR[current_state], self.nb_actions)

    def update_sr(self) -> None:
        """
        This function updates the SR given the current state-state transition matrix T.
        """
        self.SR = np.linalg.inv(np.eye(self.T.shape[0]) - self.gamma * self.T)

    def compute_update_mask(self) -> None:
        """
        This function updates the update mask.
        """
        self.update_mask = self.states.flatten(order='F') != np.tile(
            np.arange(self.nb_states), self.nb_actions
        )

    def action_probs_batch(
        self, q_function: NDArray, action_mask: None | NDArray = None
    ) -> NDArray:
        """
        This function computes the action selection probabilities
        for a table of Q-values.

        Parameters
        ----------
        q_function : NDArray
            The Q-values.
        action_mask : NDArray or None, optional
            The action mask used by the agent.

        Returns
        -------
        probs : NDArray
            The action selection probabilities.
        """
        p = np.array(
            [
                self.policy.get_action_probs(
                    q_vals, action_mask[s] if action_mask is not None else None
                )
                for s, q_vals in enumerate(q_function)
            ]
        )

        return p / np.sum(p, axis=1).reshape(p.shape[0], 1)

    def update_q(self, q_function: NDArray, update: list[Experience]) -> NDArray:
        """
        This function updates the Q-function.

        Parameters
        ----------
        q_function : NDArray
            The Q-function of the agent.
        update : list of dict
            A list containing experiences for an n-step update.

        Returns
        -------
        q_function : NDArray
            The updated Q-function.
        """
        # expected future value
        future_value = (
            np.amax(q_function[update[-1]['next_state']]) * update[-1]['terminal']
        )
        for s, step in enumerate(update):
            # sum rewards over remaining trajectory
            r = 0.0
            intermediate_terminal = 1
            for following_steps in range(len(update) - s):
                # check for intermediate terminal transitions
                if (
                    update[s + following_steps]['terminal'] == 0
                    and s != len(update) - 1
                ):
                    intermediate_terminal = 0
                    break
                r += update[s + following_steps]['reward'] * (
                    self.gamma_q**following_steps
                )
            # abort in case a terminal transition occurs within the n-step sequence
            if intermediate_terminal == 0:
                break
            # compute TD-error
            td = r + future_value * (self.gamma_q ** (following_steps + 1))
            td -= q_function[step['state']][step['action']]
            # update Q-function with TD-error
            q_function[step['state']][step['action']] += self.learning_rate_q * td

        return q_function
