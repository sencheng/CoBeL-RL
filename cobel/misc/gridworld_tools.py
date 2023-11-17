# basic imports
import numpy as np


def make_gridworld(height: int, width: int, terminals: None | list = None, rewards: None | np.ndarray = None, goals: None | list = None, starting_states: None | list = None,
                   invalid_states: None | list = None, invalid_transitions: None | list = None, wind: None | np.ndarray = None, deterministic: bool = True) -> dict:
    '''
    This function builds a gridworld according to the given parameters.
    
    Parameters
    ----------
    height :                            The height of the gridworld.
    width :                             The width of the gridworld.
    terminals :                         A list containing the gridworld's terminal states' indeces.
    rewards :                           A numpy array containing the gridworld's state rewards where the first column represents the state indeces and the second column the reward.
    goals :                             A list containing the gridworld's goals (used for visualization).
    starting_states :                   A list containing the possible starting states.
    invalid_states :                    A list containing the unreachable states.
    invalid_transitions :               A list containing the invalid state transitions.
    wind :                              The wind applied to the gridworld's states where the first column contains the state indeces and the second and thirds column the wind applied to height and width coordinates.
    deterministic :                     If true, state transition with the highest probability are chosen.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    world = dict()
    # world dimensions as integers
    world['height'] = height
    world['width'] = width
    # number of world states N
    world['states'] = height * width
    # goals for visualization as list
    world['goals'] = [] if goals is None else goals
    # terminals as arry of size N
    world['terminals'] = np.zeros(world['states'])
    if not terminals is None:
        world['terminals'][terminals] = 1
    # rewards as array of size N
    world['rewards'] = np.zeros(world['states'])
    if not rewards is None:
        world['rewards'][rewards[:, 0].astype(int)] = rewards[:, 1]
    # starting states as array of size S
    # if starting states were not defined, all states except the terminals become starting states
    world['starting_states'] = list(set([i for i in range(world['states'])]) - set([] if terminals is None else terminals))
    if starting_states is not None and len(starting_states) > 0:
        world['starting_states'] = starting_states
    world['starting_states'] = np.array(world['starting_states']).astype(int)
    # wind applied at each state as array of size Nx2
    world['wind'] = np.zeros((world['states'], 2))
    if not wind is None:
        world['wind'][wind[:, 0].astype(int)] = wind[:, 1:]
    # invalid states and transitions as lists
    world['invalid_states'] = [] if invalid_states is None else invalid_states
    world['invalid_transitions'] = [] if invalid_transitions is None else invalid_transitions
    # state coordinates as array of size Nx2
    world['coordinates'] = np.zeros((world['states'], 2))
    for i in range(width):
        for j in range(height):
            state = j * width + i
            world['coordinates'][state] = np.array([i, height - 1 - j])
    # state-action-state transitions as array of size Nx4xN
    world['sas'] = np.zeros((world['states'], 4, world['states']))
    for state in range(world['states']):
        for action in range(4):
            h = int(state/world['width'])
            w = state - h * world['width']
            # left
            if action == 0:
                w = max(0, w-1)
            # up
            elif action == 1:
                h = max(0, h-1)
            # right
            elif  action == 2:
                w = min(world['width'] - 1, w+1)
            # down
            else:
                h = min(world['height'] - 1, h+1)
            # apply wind
            # currently walls are not taken into account!
            h += world['wind'][state][0]
            w += world['wind'][state][1]
            h = min(max(0, h), world['height'] - 1)
            w = min(max(0, w), world['width'] - 1)
            # determine next state
            nextState = int(h * world['width'] + w)
            if nextState in world['invalid_states'] or (state, nextState) in world['invalid_transitions']:
                nextState = state
            world['sas'][state][action][nextState] = 1
            
    world['deterministic'] = deterministic
    
    return world
    
def make_open_field(height: int, width: int, goal_state: int = 0, reward: float = 1) -> dict:
    '''
    This function builds an open field gridworld with one terminal goal state.
    
    Parameters
    ----------
    height :                            The height of the gridworld.
    width :                             The width of the gridworld.
    goal_state :                        The goal state's index.
    reward :                            The reward provided at the goal state.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    return make_gridworld(height, width, terminals=[goal_state], rewards=np.array([[goal_state, reward]]), goals=[goal_state])

def make_empty_field(height: int, width: int) -> dict:
    '''
    This function builds an empty open field gridworld.
    
    Parameters
    ----------
    height :                            The height of the gridworld.
    width :                             The width of the gridworld.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    return make_gridworld(height, width)

def make_windy_gridworld(height: int, width: int, columns: np.ndarray, goal_state: int = 0, reward: float = 1, direction: str = 'up') -> dict:
    '''
    This function builds a windy gridworld with one terminal goal state.
    
    Parameters
    ----------
    height :                            The height of the gridworld.
    width :                             The width of the gridworld.
    columns :                           The wind strengths for the different columns.
    goal_state :                        The goal state's index.
    reward :                            The reward provided at the goal state.
    direction :                         The wind's direction (up, down).
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    directions = {'up': 1, 'down': -1}
    wind = np.zeros((height * width, 3))
    for i in range(width):
        for j in range(height):
            state = int(j * width + i)
            wind[state] = np.array([state, columns[i] * directions[direction], 0])
    return make_gridworld(height, width, terminals=[goal_state], rewards=np.array([[goal_state, reward]]), goals=[goal_state], wind=wind)

def make_t_maze(stem_length: int, arm_length: int, goal_arm: str = 'right', reward: float = 1) -> dict:
    '''
    This function builds a T-maze gridworld.
    
    Parameters
    ----------
    stem_length :                       The T-maze's stem length.
    arm_length :                        The T-maze's arm length.
    goal_arm :                          The rewarded arm (left, right).
    reward :                            The reward provided at the rewarded arm.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    # return empty dictionary if parameters are not valid
    if stem_length < 1 or arm_length < 1:
        return {}
    # compute gridworld dimensions
    height, width = stem_length + 1, arm_length * 2 + 1
    # determine goal state, reward function, terminals and starting state
    goal_state = width - 1
    if goal_arm == 'left':
        goal_state = 0
    rewards, terminals, goals = np.array([[goal_state, reward]]), [goal_state], [goal_state]
    starting_states = [height * width - arm_length - 1]
    # compute invalid transitions
    arm_states = np.arange(arm_length).reshape((arm_length, 1))
    arm_states = np.vstack((arm_states, arm_states + arm_length + 1))
    arm_states = np.hstack((arm_states, arm_states + width))
    arm_states = np.vstack((arm_states, np.flip(arm_states)))
    stem_states = np.arange(0, stem_length * width, width).reshape((stem_length, 1)) + arm_length + width - 1
    stem_states = np.vstack((stem_states, stem_states + 1))
    stem_states = np.hstack((stem_states, stem_states + 1))
    stem_states = np.vstack((stem_states, np.flip(stem_states)))
    invalid_transitions = list(map(tuple, np.vstack((arm_states, stem_states))))
    
    return make_gridworld(height, width, terminals, rewards, goals, starting_states, invalid_transitions=invalid_transitions)

def make_double_t_maze(stem_length: int, arm_length: int, goal_arm: str = 'right-right', reward: float = 1) -> dict:
    '''
    This function builds a double T-maze gridworld.
    
    Parameters
    ----------
    stem_length :                       The double T-maze's stem length.
    arm_length :                        The double T-maze's arm length.
    goal_arm :                          The rewarded arm (left-left, left-right, right-left, right-right).
    reward :                            The reward provided at the rewarded arm.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    # return empty dictionary if parameters are not valid
    if stem_length < 1 or arm_length < 1:
        return {}
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
    rewards, terminals, goals = np.array([[goal_state, reward]]), [goal_state], [goal_state]
    starting_states = [height * width - arm_length * 2 - 2]
    # compute invalid transitions
    arm_states = np.arange(arm_length).reshape((arm_length, 1))
    arm_states = np.vstack((arm_states, arm_states + arm_length + 1,
                            arm_states + arm_length * 2 + 2, arm_states + arm_length * 3 + 3))
    arm_states = np.vstack((arm_states, np.arange(arm_length * 2 + 1).reshape((arm_length * 2 + 1, 1)) + arm_length + 1 + width * stem_length))
    arm_states = np.vstack((arm_states, np.arange(arm_length + 1).reshape((arm_length + 1, 1)) + arm_length + width * (stem_length + 1)))
    arm_states = np.vstack((arm_states, np.arange(arm_length + 1).reshape((arm_length + 1, 1)) + arm_length * 2 + 2 + width * (stem_length + 1)))
    arm_states = np.hstack((arm_states, arm_states + width))
    arm_states = np.vstack((arm_states, np.flip(arm_states)))
    stem_states = np.arange(0, stem_length * width, width).reshape((stem_length, 1))
    stem_states = np.vstack((stem_states + width + arm_length,
                             stem_states + width + arm_length * 3 + 1,
                             stem_states + width * (stem_length + 2) + arm_length * 2,
                             stem_states + width * (stem_length + 2) + arm_length * 2 + 1))
    stem_states = np.vstack((stem_states, np.arange(0, stem_length * (width + 1), width).reshape((stem_length + 1, 1)) + arm_length + width - 1))
    stem_states = np.vstack((stem_states, np.arange(0, stem_length * (width + 1), width).reshape((stem_length + 1, 1)) + arm_length * 3 + width + 2))
    stem_states = np.hstack((stem_states, stem_states + 1))
    stem_states = np.vstack((stem_states, np.array([[arm_length * 2, arm_length * 2 + 1], [arm_length * 3, arm_length * 2 + 1]])))
    stem_states = np.vstack((stem_states, np.flip(stem_states)))
    invalid_transitions = list(map(tuple, np.vstack((arm_states, stem_states))))
    
    return make_gridworld(height, width, terminals, rewards, goals, starting_states, invalid_transitions=invalid_transitions)

def make_two_sided_t_maze(stem_length: int, arm_length: int, goal_arm: str = 'right-right', reward: float = 1) -> dict:
    '''
    This function builds a two sided T-maze gridworld.
    
    Parameters
    ----------
    stem_length :                       The two sided T-maze's stem length.
    arm_length :                        The two sided T-maze's arm length.
    goal_arm :                          The rewarded arm (left-left, left-right, right-left, right-right).
    reward :                            The reward provided at the rewarded arm.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    # return empty dictionary if parameters are not valid
    if stem_length < 1 or arm_length < 1:
        return {}
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
    rewards, terminals, goals = np.array([[goal_state, reward]]), [goal_state], [goal_state]
    starting_states = [arm_length * width + int(stem_length/2)]
    # compute invalid transitions
    arm_states = np.arange(0, arm_length * width, width).reshape((arm_length, 1))
    arm_states = np.vstack((arm_states, arm_states + width * (arm_length + 1),
                            arm_states + width - 2, arm_states + width * (arm_length + 1) + width -2))
    arm_states = np.hstack((arm_states, arm_states + 1))
    arm_states = np.vstack((arm_states, np.flip(arm_states)))
    stem_states = np.arange(stem_length).reshape((stem_length, 1)) + width * (arm_length - 1) +  1
    stem_states = np.vstack((stem_states, stem_states + width))
    stem_states = np.hstack((stem_states, stem_states + width))
    stem_states = np.vstack((stem_states, np.flip(stem_states)))
    invalid_transitions = list(map(tuple, np.vstack((arm_states, stem_states))))
    
    return make_gridworld(height, width, terminals, rewards, goals, starting_states, invalid_transitions=invalid_transitions)

def make_two_choice_t_maze(center_height: int, lap_width: int, arm_length: int, chirality: str = 'right', goal_location: str = 'right', reward: float = 1) -> dict:
    '''
    This function builds a two-choice T-maze gridworld.
    
    Parameters
    ----------
    center_height :                     The height of the maze's center piece.
    lap_width :                         The width of the laps.
    arm_length :                        The length of the (inner) T-maze's arms.
    chirality :                         Defines whether the T-maze is located at the left or right side of the maze.
    goal_location :                     The rewarded lap (left, right).
    reward :                            The reward provided at the rewarded arm.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    # compute gridworld dimensions
    height, width = center_height + 2, lap_width * 2 + 3
    # ! to be implemented !

    return make_open_field(height, width, 0, reward)

def make_8_maze(center_height: int, lap_width: int, goal_location: str = 'right', reward: float = 1) -> dict:
    '''
    This function builds an 8-maze gridworld.
    
    Parameters
    ----------
    center_height :                     The height of the maze's center piece.
    lap_width :                         The width of the laps.
    goal_location :                     The rewarded lap (left, right).
    reward :                            The reward provided at the rewarded arm.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    # return empty dictionary if parameters are not valid
    if center_height < 1 or lap_width < 1:
        return {}
    # compute gridworld dimensions
    height, width = center_height + 2, lap_width * 2 + 3
    # determine goal state, reward function, terminals and starting state
    goal_state = int((center_height + 2)/2) * width
    if goal_location == 'right':
        goal_state += width - 1
    rewards, terminals, goals = np.array([[goal_state, reward]]), [goal_state], [goal_state]
    starting_states = [lap_width + 2]
    # compute invalid transitions
    horizontal = np.arange(0, lap_width).reshape(lap_width, 1) + 1
    horizontal = np.vstack((horizontal, horizontal + lap_width + 1))
    horizontal = np.vstack((horizontal, horizontal + width * (height - 2)))
    horizontal = np.hstack((horizontal, horizontal + width))
    horizontal = np.vstack((horizontal, np.flip(horizontal)))
    vertical = np.arange(0, center_height * width, width).reshape((center_height, 1)) +  width
    vertical = np.vstack((vertical, vertical + lap_width, vertical + lap_width + 1, vertical + lap_width * 2 + 1))
    vertical = np.hstack((vertical, vertical + 1))
    vertical = np.vstack((vertical, np.flip(vertical)))
    invalid_transitions = list(map(tuple, np.vstack((horizontal, vertical))))
    
    return make_gridworld(height, width, terminals, rewards, goals, starting_states, invalid_transitions=invalid_transitions)

def make_detour_maze(width_small: int, height_small: int, width_large: int, height_large: int, reward: float = 1) -> dict:
    '''
    This function builds a detour maze gridworld.
    
    Parameters
    ----------
    width_small :                       The width of the detour maze's small side piece.
    height_small :                      The height of the detour maze's small side piece.
    width_large :                       The width of the detour maze's large side piece.
    height_large :                      The height of the detour maze's large side piece.
    reward :                            The reward provided at the rewarded arm.
    
    Returns
    ----------
    world :                             The gridworld as a dictionary.
    '''
    # return empty dictionary if parameters are not valid
    if width_small < 1 or height_small < 1 or width_small >= width_large or height_small >= height_large:
        return {}
    # compute gridworld dimensions
    width, height = width_small + width_large + 3, height_small + height_large + 5
    # determine goal state, reward function, terminals and starting state
    goal_state = width_small + 1
    rewards, terminals, goals = np.array([[goal_state, reward]]), [goal_state], [goal_state]
    starting_states = [width_small + 1 + width * (height - 1)]
    # compute invalid transitions
    horizontal = np.arange(0, width_small + 1).reshape(width_small + 1, 1) + width * (height_large + 1)
    horizontal = np.vstack((horizontal, horizontal + width * (height_small + 2)))
    horizontal = np.vstack((horizontal, np.arange(0, width_small).reshape(width_small, 1) + width * (height_large + 2) + 1))
    horizontal = np.vstack((horizontal, np.arange(0, width_small).reshape(width_small, 1) + width * (height_large + height_small + 2) + 1))
    horizontal = np.vstack((horizontal, np.arange(0, width_large + 1).reshape(width_large + 1, 1) + width_small + 2))
    horizontal = np.vstack((horizontal, np.arange(0, width_large + 1).reshape(width_large + 1, 1) + width * (height_large + 2) + width_small + 2))
    horizontal = np.vstack((horizontal, np.arange(0, width_large).reshape(width_large, 1) + width + width_small + 2))
    horizontal = np.vstack((horizontal, np.arange(0, width_large).reshape(width_large, 1) + width * (height_large + 1) + width_small + 2))
    horizontal = np.hstack((horizontal, horizontal + width))
    horizontal = np.vstack((horizontal, np.flip(horizontal)))
    vertical = np.arange(0, (height_large + 2) * width, width).reshape(height_large + 2, 1) + width_small
    vertical = np.vstack((vertical, np.array([[width_small + 1], [width_small + width * (height - 1)]])))
    vertical = np.vstack((vertical, np.arange(0, width * (height_small + 2), width).reshape(height_small + 2, 1) + width * (height_large + 3) + width_small + 1))
    vertical = np.vstack((vertical, np.arange(0, height_small * width, width).reshape(height_small, 1) + width * (height_large + 3)))
    vertical = np.vstack((vertical, np.arange(0, height_small * width, width).reshape(height_small, 1) + width * (height_large + 3) + width_small))
    vertical = np.vstack((vertical, np.arange(0, height_large * width, width).reshape(height_large, 1) + width * 2 + width_small + 1))
    vertical = np.vstack((vertical, np.arange(0, height_large * width, width).reshape(height_large, 1) + width * 2 + width_small + width_large + 1))
    vertical = np.hstack((vertical, vertical + 1))
    vertical = np.vstack((vertical, np.flip(vertical)))
    invalid_transitions = list(map(tuple, np.vstack((horizontal, vertical))))
    
    return make_gridworld(height, width, terminals, rewards, goals, starting_states, invalid_transitions=invalid_transitions)

if __name__ == '__main__':
    height, width = 5, 5
    goal = 4
    reward = 1
    columns = np.array([0, 0, 0, 1, 1])
    gridworld = make_gridworld(height, width, terminals=[goal], rewards=np.array([[goal, reward]]))
    open_field = make_open_field(height, width, goal, reward)
    windy_gridworld = make_windy_gridworld(height, width, columns, goal, reward)
