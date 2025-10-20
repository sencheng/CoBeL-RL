# basic imports
import numpy as np
# typing
from typing import Literal
from numpy.typing import NDArray
from ..interface.gridworld import WorldDict


def make_gridworld(
    height: int,
    width: int,
    terminals: None | list[int] = None,
    rewards: None | np.ndarray = None,
    goals: None | list[int] = None,
    starting_states: None | list[int] = None,
    invalid_states: None | list[int] = None,
    invalid_transitions: None | list[tuple[int, int]] = None,
    wind: None | np.ndarray = None,
    deterministic: bool = True,
) -> WorldDict:
    """
    This function builds a gridworld according to the given parameters.

    Parameters
    ----------
    height : int
        The height of the gridworld.
    width : int
        The width of the gridworld.
    terminals : list of int or None, optional
        A list containing the gridworld's terminal states' indeces.
    rewards : NDArray or None, optional
        A numpy array containing the gridworld's state rewards
        where the first column represents the state indeces and
        the second column the reward.
    goals : list of int or None, optional
        A list containing the gridworld's goals (used for visualization).
    starting_states : list of int or None, optional
        A list containing the possible starting states.
    invalid_states : list of int or None, optional
        A list containing the unreachable states.
    invalid_transitions : list of 2-tuple of int or None, optional
        A list containing the invalid state transitions.
    wind : NDArray or None, optional
        The wind applied to the gridworld's states where the first
        column contains the state indeces and the second and thirds
        column the wind applied to height and width coordinates.
    deterministic : bool, default=True
        If true, state transition with the highest probability are chosen.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    world: WorldDict = {}  # type: ignore
    # world dimensions as integers
    world['height'] = height
    world['width'] = width
    # number of world states N
    world['states'] = height * width
    # goals for visualization as list
    world['goals'] = [] if goals is None else goals
    # terminals as arry of size N
    world['terminals'] = np.zeros(world['states']).astype(int)
    if terminals is not None:
        world['terminals'][terminals] = 1
    # rewards as array of size N
    world['rewards'] = np.zeros(world['states']).astype(float)
    if rewards is not None:
        world['rewards'][rewards[:, 0].astype(int)] = rewards[:, 1]
    # starting states as array of size S
    # if starting states were not defined,
    # all states except the terminals become starting states
    state_set = set([i for i in range(int(world['states']))])
    terminal_set = set([] if terminals is None else terminals)
    default_starting_states = list(state_set - terminal_set)
    if starting_states is not None and len(starting_states) > 0:
        world['starting_states'] = np.array(starting_states)
    else:
        world['starting_states'] = np.array(default_starting_states)
    # wind applied at each state as array of size Nx2
    world['wind'] = np.zeros((world['states'], 2)).astype(int)
    if wind is not None:
        world['wind'][wind[:, 0].astype(int)] = wind[:, 1:].astype(int)
    # invalid states and transitions as lists
    if invalid_states is None:
        world['invalid_states'] = []
    else:
        world['invalid_states'] = invalid_states
    if invalid_transitions is None:
        world['invalid_transitions'] = []
    else:
        world['invalid_transitions'] = invalid_transitions
    # state coordinates as array of size Nx2
    world['coordinates'] = np.zeros((world['states'], 2)).astype(float)
    for i in range(width):
        for j in range(height):
            state = j * width + i
            world['coordinates'][state] = np.array([i, height - 1 - j]).astype(float)
    # state-action-state transitions as array of size Nx4xN
    world['sas'] = np.zeros((world['states'], 4, world['states']))
    for state in range(world['states']):
        for action in range(4):
            h = int(state / world['width'])
            w = state - h * world['width']
            # left
            if action == 0:
                w = max(0, w - 1)
            # up
            elif action == 1:
                h = max(0, h - 1)
            # right
            elif action == 2:
                w = min(world['width'] - 1, w + 1)
            # down
            else:
                h = min(world['height'] - 1, h + 1)
            # apply wind
            # currently walls are not taken into account!
            h += world['wind'][state][0]
            w += world['wind'][state][1]
            h = min(max(0, h), world['height'] - 1)
            w = min(max(0, w), world['width'] - 1)
            # determine next state
            next_state = int(h * world['width'] + w)
            if (
                next_state in world['invalid_states']
                or (state, next_state) in world['invalid_transitions']
            ):
                next_state = state
            world['sas'][state][action][next_state] = 1
    world['deterministic'] = deterministic

    return world


def make_open_field(
    height: int, width: int, goal_state: int = 0, reward: float = 1
) -> WorldDict:
    """
    This function builds an open field gridworld with one terminal goal state.

    Parameters
    ----------
    height : int
        The height of the gridworld.
    width : int
        The width of the gridworld.
    goal_state : int, default=0
        The goal state's index.
    reward : float, default=1.
        The reward provided at the goal state.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    return make_gridworld(
        height,
        width,
        terminals=[goal_state],
        rewards=np.array([[goal_state, reward]]),
        goals=[goal_state],
    )


def make_empty_field(height: int, width: int) -> WorldDict:
    """
    This function builds an empty open field gridworld.

    Parameters
    ----------
    height : int
        The height of the gridworld.
    width : int
        The width of the gridworld.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    return make_gridworld(height, width)


def make_windy_gridworld(
    height: int,
    width: int,
    columns: np.ndarray,
    goal_state: int = 0,
    reward: float = 1,
    direction: Literal['up', 'down'] = 'up',
) -> WorldDict:
    """
    This function builds a windy gridworld with one terminal goal state.

    Parameters
    ----------
    height : int
        The height of the gridworld.
    width : int
        The width of the gridworld.
    columns : NDArray
        The wind strengths for the different columns.
    goal_state : int, default=0
        The goal state's index.
    reward : float, default=1.
        The reward provided at the goal state.
    direction : str, default='up'
        The wind's direction (up, down).

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    directions = {'up': 1, 'down': -1}
    wind = np.zeros((height * width, 3))
    for i in range(width):
        for j in range(height):
            state = int(j * width + i)
            wind[state] = np.array([state, columns[i] * directions[direction], 0])

    return make_gridworld(
        height,
        width,
        terminals=[goal_state],
        rewards=np.array([[goal_state, reward]]),
        goals=[goal_state],
        wind=wind,
    )


def make_t_maze(
    stem_length: int,
    arm_length: int,
    goal_arm: Literal['left', 'right'] = 'right',
    reward: float = 1,
) -> WorldDict:
    """
    This function builds a T-maze gridworld.

    Parameters
    ----------
    stem_length : int
        The T-maze's stem length.
    arm_length : int
        The T-maze's arm length.
    goal_arm : str, default='right'
        The rewarded arm (left, right).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    assert stem_length > 0 and arm_length > 0, (
        'Stem and arm length must be greater than zero!'
    )
    # compute gridworld dimensions
    height, width = stem_length + 1, arm_length * 2 + 1
    # determine goal state, reward function, terminals and starting state
    goal_state = width - 1
    if goal_arm == 'left':
        goal_state = 0
    rewards, terminals = np.array([[goal_state, reward]]), [goal_state]
    goals = [goal_state]
    starting_states = [height * width - arm_length - 1]
    # compute invalid transitions
    arm_states = np.arange(arm_length).reshape((arm_length, 1))
    arm_states = np.vstack((arm_states, arm_states + arm_length + 1))
    arm_states = np.hstack((arm_states, arm_states + width))
    arm_states = np.vstack((arm_states, np.flip(arm_states)))
    stem_states = (
        np.arange(0, stem_length * width, width).reshape((stem_length, 1))
        + arm_length
        + width
        - 1
    )
    stem_states = np.vstack((stem_states, stem_states + 1))
    stem_states = np.hstack((stem_states, stem_states + 1))
    stem_states = np.vstack((stem_states, np.flip(stem_states)))
    invalid_transitions = list(map(tuple, np.vstack((arm_states, stem_states))))

    return make_gridworld(
        height,
        width,
        terminals,
        rewards,
        goals,
        starting_states,
        invalid_transitions=invalid_transitions,
    )


def make_double_t_maze(
    stem_length: int,
    arm_length: int,
    goal_arm: Literal[
        'left-left', 'left-right', 'right-left', 'right-right'
    ] = 'right-right',
    reward: float = 1,
) -> WorldDict:
    """
    This function builds a double T-maze gridworld.

    Parameters
    ----------
    stem_length : int
        The double T-maze's stem length.
    arm_length : int
        The double T-maze's arm length.
    goal_arm : str, default='right-right'
        The rewarded arm (left-left, left-right, right-left, right-right).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    assert stem_length > 0 and arm_length > 0, (
        'Stem and arm length must be greater than zero!'
    )
    # compute gridworld dimensions
    height, width = stem_length * 2 + 2, arm_length * 4 + 3
    # determine goal state, reward function, terminals and starting state
    goal_state = 0
    if goal_arm == 'left-right':
        goal_state = arm_length * 2
    elif goal_arm == 'right-left':
        goal_state = arm_length * 2 + 2
    elif goal_arm == 'right-right':
        goal_state = arm_length * 4 + 2
    rewards, terminals = np.array([[goal_state, reward]]), [goal_state]
    goals = [goal_state]
    starting_states = [height * width - arm_length * 2 - 2]
    # compute invalid transitions
    arm_states = np.arange(arm_length).reshape((arm_length, 1))
    arm_states = np.vstack(
        (
            arm_states,
            arm_states + arm_length + 1,
            arm_states + arm_length * 2 + 2,
            arm_states + arm_length * 3 + 3,
        )
    )
    arm_states = np.vstack(
        (
            arm_states,
            np.arange(arm_length * 2 + 1).reshape((arm_length * 2 + 1, 1))
            + arm_length
            + 1
            + width * stem_length,
        )
    )
    arm_states = np.vstack(
        (
            arm_states,
            np.arange(arm_length + 1).reshape((arm_length + 1, 1))
            + arm_length
            + width * (stem_length + 1),
        )
    )
    arm_states = np.vstack(
        (
            arm_states,
            np.arange(arm_length + 1).reshape((arm_length + 1, 1))
            + arm_length * 2
            + 2
            + width * (stem_length + 1),
        )
    )
    arm_states = np.hstack((arm_states, arm_states + width))
    arm_states = np.vstack((arm_states, np.flip(arm_states)))
    stem_states = np.arange(0, stem_length * width, width).reshape((stem_length, 1))
    stem_states = np.vstack(
        (
            stem_states + width + arm_length,
            stem_states + width + arm_length * 3 + 1,
            stem_states + width * (stem_length + 2) + arm_length * 2,
            stem_states + width * (stem_length + 2) + arm_length * 2 + 1,
        )
    )
    stem_states = np.vstack(
        (
            stem_states,
            np.arange(0, stem_length * (width + 1), width).reshape((stem_length + 1, 1))
            + arm_length
            + width
            - 1,
        )
    )
    stem_states = np.vstack(
        (
            stem_states,
            np.arange(0, stem_length * (width + 1), width).reshape((stem_length + 1, 1))
            + arm_length * 3
            + width
            + 2,
        )
    )
    stem_states = np.hstack((stem_states, stem_states + 1))
    stem_states = np.vstack(
        (
            stem_states,
            np.array(
                [
                    [arm_length * 2, arm_length * 2 + 1],
                    [arm_length * 2 + 1, arm_length * 2 + 2],
                ]
            ),
        )
    )
    stem_states = np.vstack((stem_states, np.flip(stem_states)))
    invalid_transitions = list(map(tuple, np.vstack((arm_states, stem_states))))

    return make_gridworld(
        height,
        width,
        terminals,
        rewards,
        goals,
        starting_states,
        invalid_transitions=invalid_transitions,
    )


def make_two_sided_t_maze(
    stem_length: int,
    arm_length: int,
    goal_arm: Literal[
        'left-left', 'left-right', 'right-left', 'right-right'
    ] = 'right-right',
    reward: float = 1,
) -> WorldDict:
    """
    This function builds a two sided T-maze gridworld.

    Parameters
    ----------
    stem_length : int
        The two sided T-maze's stem length.
    arm_length : int
        The two sided T-maze's arm length.
    goal_arm : str, default='right-right'
        The rewarded arm (left-left, left-right, right-left, right-right).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    assert stem_length > 0 and arm_length > 0, (
        'Stem and arm length must be greater than zero!'
    )
    # compute gridworld dimensions
    height, width = arm_length * 2 + 1, stem_length + 2
    # determine goal state, reward function, terminals and starting state
    goal_state = 0
    if goal_arm == 'right-left':
        goal_state = width - 1
    elif goal_arm == 'left-left':
        goal_state = width * (height - 1)
    elif goal_arm == 'right-right':
        goal_state = width * height - 1
    rewards, terminals = np.array([[goal_state, reward]]), [goal_state]
    goals = [goal_state]
    starting_states = [arm_length * width + int(stem_length / 2)]
    # compute invalid transitions
    arm_states = np.arange(0, arm_length * width, width).reshape((arm_length, 1))
    arm_states = np.vstack(
        (
            arm_states,
            arm_states + width * (arm_length + 1),
            arm_states + width - 2,
            arm_states + width * (arm_length + 1) + width - 2,
        )
    )
    arm_states = np.hstack((arm_states, arm_states + 1))
    arm_states = np.vstack((arm_states, np.flip(arm_states)))
    stem_states = (
        np.arange(stem_length).reshape((stem_length, 1)) + width * (arm_length - 1) + 1
    )
    stem_states = np.vstack((stem_states, stem_states + width))
    stem_states = np.hstack((stem_states, stem_states + width))
    stem_states = np.vstack((stem_states, np.flip(stem_states)))
    invalid_transitions = list(map(tuple, np.vstack((arm_states, stem_states))))

    return make_gridworld(
        height,
        width,
        terminals,
        rewards,
        goals,
        starting_states,
        invalid_transitions=invalid_transitions,
    )


def make_two_choice_t_maze(
    center_height: int,
    lap_width: int,
    arm_length: int,
    chirality: Literal['left', 'right'] = 'right',
    goal_location: Literal['left', 'right'] = 'right',
    reward: float = 1,
) -> WorldDict:
    """
    This function builds a two-choice T-maze gridworld.

    Parameters
    ----------
    center_height : int
        The height of the maze's center piece.
    lap_width : int
        The width of the laps.
    arm_length : int
        The length of the (inner) T-maze's arms.
    chirality : str, default='right'
        Defines whether the T-maze is located at the left or right side of the maze.
    goal_location : str, default='right'
        The rewarded lap (left, right).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    assert arm_length > 0, '!'
    assert center_height > 2, '!'
    assert lap_width >= arm_length * 2 + 1, '!'
    assert chirality in ['left', 'right'], 'Invalid chirality!'
    # compute gridworld dimensions
    height, width = center_height + 2, lap_width * 2 + 3
    states = height * width
    # prepare gridworld info
    rewards = np.zeros((states, 2))
    rewards[:, 0] = np.arange(states)
    goal_state = int(height / 2) * width
    starting_state = width * (height - 1) + lap_width + arm_length
    if goal_location == 'right':
        goal_state += width - 1
    rewards[goal_state, 1] = 1.0
    goals, starting_states, terminals = [goal_state], [starting_state], [goal_state]
    # compute gridworld barriers
    borders_h = np.arange(lap_width).reshape((lap_width, 1)) + 1
    borders_h = np.vstack((borders_h, borders_h + lap_width + 1))
    borders_v = (
        np.arange(0, width * center_height, width).reshape((center_height, 1)) + width
    )
    borders_v = np.vstack((borders_v, borders_v + width - 2))
    center_top_v = (
        np.arange(0, width * int((center_height - 1) / 2), width).reshape(
            (int((center_height - 1) / 2), 1)
        )
        + width
        + lap_width
    )
    center_top_v = np.vstack((center_top_v, center_top_v + 1))
    borders_v = np.vstack((borders_v, center_top_v))
    center_mid_v = np.array([[lap_width - arm_length * 2], [lap_width + 1]]) + width * (
        int((center_height - 1) / 2) + 1
    )
    center_bot_v = (
        np.arange(0, width * int(center_height / 2), width).reshape(
            (int(center_height / 2), 1)
        )
        + width * (int((center_height - 1) / 2) + 2)
        + lap_width
        - arm_length
    )
    center_bot_v = np.vstack((center_bot_v, center_bot_v + 1))
    center_mid_h = (
        np.arange(2 * arm_length).reshape((2 * arm_length, 1))
        + width * int((center_height - 1) / 2)
        + lap_width
        - arm_length * 2
        + 1
    )
    center_mid_h = np.vstack(
        (
            center_mid_h,
            np.arange(arm_length).reshape((arm_length, 1))
            + width * (int((center_height - 1) / 2) + 1)
            + lap_width
            - arm_length * 2
            + 1,
        )
    )
    center_mid_h = np.vstack(
        (
            center_mid_h,
            np.arange(arm_length).reshape((arm_length, 1))
            + width * (int((center_height - 1) / 2) + 1)
            + lap_width
            - arm_length
            + 2,
        )
    )
    center_bot_h: NDArray
    if chirality == 'left':
        center_bot_h = (
            np.arange(lap_width - arm_length).reshape((lap_width - arm_length, 1))
            + width * (height - 2)
            + 1
        )
        center_bot_h = np.vstack(
            (
                center_bot_h,
                np.arange(lap_width + arm_length).reshape((lap_width + arm_length, 1))
                + width * (height - 2)
                + lap_width
                + 2
                - arm_length,
            )
        )
    else:
        center_mid_h += 2 * arm_length
        center_mid_v += 2 * arm_length
        center_bot_v += 2 * arm_length
        center_mid_h[: (2 * arm_length)] += 1
        center_bot_h = (
            np.arange(lap_width + arm_length).reshape((lap_width + arm_length, 1))
            + width * (height - 2)
            + 1
        )
        center_bot_h = np.vstack(
            (
                center_bot_h,
                np.arange(lap_width - arm_length).reshape((lap_width - arm_length, 1))
                + width * (height - 2)
                + lap_width
                + 2
                + arm_length,
            )
        )
    borders_v = np.vstack((borders_v, center_mid_v))
    borders_v = np.vstack((borders_v, center_bot_v))
    borders_h = np.vstack((borders_h, center_mid_h))
    borders_h = np.vstack((borders_h, center_bot_h))
    borders_h = np.hstack((borders_h, borders_h + width))
    borders_v = np.hstack((borders_v, borders_v + 1))
    borders = np.vstack((borders_h, borders_v))
    borders = np.vstack((borders, np.flip(borders)))
    invalid_transitions = list(map(tuple, borders))

    return make_gridworld(
        height,
        width,
        terminals,
        rewards,
        goals,
        starting_states,
        invalid_transitions=invalid_transitions,
    )


def make_8_maze(
    center_height: int,
    lap_width: int,
    goal_location: Literal['left', 'right'] = 'right',
    reward: float = 1,
) -> WorldDict:
    """
    This function builds an 8-maze gridworld.

    Parameters
    ----------
    center_height : int
        The height of the maze's center piece.
    lap_width : int
        The width of the laps.
    goal_location : str, default='right'
        The rewarded lap (left, right).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    assert center_height > 0 and lap_width > 0, (
        'Center height and lap width must be greater than zero!'
    )
    # compute gridworld dimensions
    height, width = center_height + 2, lap_width * 2 + 3
    # determine goal state, reward function, terminals and starting state
    goal_state = int((center_height + 2) / 2) * width
    if goal_location == 'right':
        goal_state += width - 1
    rewards, terminals = np.array([[goal_state, reward]]), [goal_state]
    goals = [goal_state]
    starting_states = [lap_width + 2]
    # compute invalid transitions
    horizontal = np.arange(0, lap_width).reshape(lap_width, 1) + 1
    horizontal = np.vstack((horizontal, horizontal + lap_width + 1))
    horizontal = np.vstack((horizontal, horizontal + width * (height - 2)))
    horizontal = np.hstack((horizontal, horizontal + width))
    horizontal = np.vstack((horizontal, np.flip(horizontal)))
    vertical = (
        np.arange(0, center_height * width, width).reshape((center_height, 1)) + width
    )
    vertical = np.vstack(
        (
            vertical,
            vertical + lap_width,
            vertical + lap_width + 1,
            vertical + lap_width * 2 + 1,
        )
    )
    vertical = np.hstack((vertical, vertical + 1))
    vertical = np.vstack((vertical, np.flip(vertical)))
    invalid_transitions = list(map(tuple, np.vstack((horizontal, vertical))))

    return make_gridworld(
        height,
        width,
        terminals,
        rewards,
        goals,
        starting_states,
        invalid_transitions=invalid_transitions,
    )


def make_detour_maze(
    width_small: int,
    height_small: int,
    width_large: int,
    height_large: int,
    reward: float = 1,
) -> WorldDict:
    """
    This function builds a detour maze gridworld.

    Parameters
    ----------
    width_small : int
        The width of the detour maze's small side piece.
    height_small : int
        The height of the detour maze's small side piece.
    width_large : int
        The width of the detour maze's large side piece.
    height_large : int
        The height of the detour maze's large side piece.
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    assert width_small > 0 and height_small > 0, (
        'Width and height of the small side piece must be greater than zero!'
    )
    assert width_large > width_small and height_large > height_small, (
        'Width and height of the large side piece must'
        ' be greater than those of the small side piece!'
    )
    # compute gridworld dimensions
    width, height = width_small + width_large + 3, height_small + height_large + 5
    # determine goal state, reward function, terminals and starting state
    goal_state = width_small + 1
    rewards, terminals = np.array([[goal_state, reward]]), [goal_state]
    goals = [goal_state]
    starting_states = [width_small + 1 + width * (height - 1)]
    # compute invalid transitions
    horizontal = np.arange(0, width_small + 1).reshape(width_small + 1, 1) + width * (
        height_large + 1
    )
    horizontal = np.vstack((horizontal, horizontal + width * (height_small + 2)))
    horizontal = np.vstack(
        (
            horizontal,
            np.arange(0, width_small).reshape(width_small, 1)
            + width * (height_large + 2)
            + 1,
        )
    )
    horizontal = np.vstack(
        (
            horizontal,
            np.arange(0, width_small).reshape(width_small, 1)
            + width * (height_large + height_small + 2)
            + 1,
        )
    )
    horizontal = np.vstack(
        (
            horizontal,
            np.arange(0, width_large + 1).reshape(width_large + 1, 1) + width_small + 2,
        )
    )
    horizontal = np.vstack(
        (
            horizontal,
            np.arange(0, width_large + 1).reshape(width_large + 1, 1)
            + width * (height_large + 2)
            + width_small
            + 2,
        )
    )
    horizontal = np.vstack(
        (
            horizontal,
            np.arange(0, width_large).reshape(width_large, 1) + width + width_small + 2,
        )
    )
    horizontal = np.vstack(
        (
            horizontal,
            np.arange(0, width_large).reshape(width_large, 1)
            + width * (height_large + 1)
            + width_small
            + 2,
        )
    )
    horizontal = np.hstack((horizontal, horizontal + width))
    horizontal = np.vstack((horizontal, np.flip(horizontal)))
    vertical = (
        np.arange(0, (height_large + 2) * width, width).reshape(height_large + 2, 1)
        + width_small
    )
    vertical = np.vstack(
        (vertical, np.array([[width_small + 1], [width_small + width * (height - 1)]]))
    )
    vertical = np.vstack(
        (
            vertical,
            np.arange(0, width * (height_small + 2), width).reshape(height_small + 2, 1)
            + width * (height_large + 3)
            + width_small
            + 1,
        )
    )
    vertical = np.vstack(
        (
            vertical,
            np.arange(0, height_small * width, width).reshape(height_small, 1)
            + width * (height_large + 3),
        )
    )
    vertical = np.vstack(
        (
            vertical,
            np.arange(0, height_small * width, width).reshape(height_small, 1)
            + width * (height_large + 3)
            + width_small,
        )
    )
    vertical = np.vstack(
        (
            vertical,
            np.arange(0, height_large * width, width).reshape(height_large, 1)
            + width * 2
            + width_small
            + 1,
        )
    )
    vertical = np.vstack(
        (
            vertical,
            np.arange(0, height_large * width, width).reshape(height_large, 1)
            + width * 2
            + width_small
            + width_large
            + 1,
        )
    )
    vertical = np.hstack((vertical, vertical + 1))
    vertical = np.vstack((vertical, np.flip(vertical)))
    invalid_transitions = list(map(tuple, np.vstack((horizontal, vertical))))

    return make_gridworld(
        height,
        width,
        terminals,
        rewards,
        goals,
        starting_states,
        invalid_transitions=invalid_transitions,
    )


def make_cross_maze(
    arm_length: int,
    arm_width: int,
    goal_arm: Literal['left', 'top', 'right', 'bottom'] = 'top',
    reward: float = 1.0,
) -> WorldDict:
    """
    This function builds a cross maze gridworld.

    Parameters
    ----------
    arm_length : int
        The cross maze's arm length.
    arm_width : int
        The cross maze's arm width.
    goal_arm : str, default='top'
        The rewarded arm (left, top, right, bottom).
    reward : float, default=1.
        The reward provided at the rewarded arm.

    Returns
    -------
    world : WorldDict
        The gridworld as a dictionary.
    """
    assert arm_length > 0, 'The arm length must be greater than zero!'
    assert arm_width > 0, 'The arm width must be greater than zero!'
    assert goal_arm in ('left', 'top', 'right', 'bottom'), 'Invalid goal arm!'
    size = arm_length * 2 + arm_width
    # horizontal
    states = np.arange(arm_length) + (arm_length - 1) * size
    states = np.concatenate(
        (
            states,
            states + arm_width + arm_length,
            states + arm_width * size,
            states + arm_width * size + arm_width + arm_length,
        )
    )
    states_r = states.reshape((states.size, 1))
    trans = np.hstack((states_r, states_r + size))
    trans = np.vstack((trans, np.flip(trans)))
    invalid_transitions: list[tuple[int, int]] = list(map(tuple, trans))
    # vertical
    states = np.arange(arm_length) * size + arm_length
    states = np.concatenate((states, states + arm_width))
    states = np.concatenate((states, states + (arm_width + arm_length) * size))
    states_r = states.reshape((states.size, 1))
    trans = np.hstack((states_r, states_r - 1))
    trans = np.vstack((trans, np.flip(trans)))
    invalid_transitions += list(map(tuple, trans))
    # starting states
    starting: list[int] = []
    offset = arm_length * size + arm_length
    for i in range(arm_width):
        for j in range(arm_width):
            starting.append(offset + i * size + j)
    # rewards and terminals
    terminals: list[int] = list(np.arange(arm_width) + arm_length)
    if goal_arm == 'left':
        terminals = list(np.arange(arm_width) * size + arm_length * size)
    elif goal_arm == 'right':
        terminals = list(np.arange(arm_width) * size + arm_length * size + (size - 1))
    elif goal_arm == 'bottom':
        terminals = list(np.arange(arm_width) + arm_length + size * (size - 1))
    rewards = np.ones((arm_width, 2))
    rewards[:, 0] = terminals

    return make_gridworld(
        size,
        size,
        terminals,
        rewards,
        terminals,
        starting_states=starting,
        invalid_transitions=invalid_transitions,
    )
