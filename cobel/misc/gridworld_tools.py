# basic imports
import numpy as np


def make_gridworld(height: int, width: int, terminals: list = [], rewards: np.ndarray = None, goals: list = [], starting_states: list = [],
                   invalid_states: list = [], invalid_transitions: list = [], wind: np.ndarray = None, deterministic: bool = True) -> dict:
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
    world['goals'] = goals
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
    world['starting_states'] = list(set([i for i in range(world['states'])]) - set(terminals))
    if len(starting_states) > 0:
        world['starting_states'] = starting_states
    world['starting_states'] = np.array(world['starting_states']).astype(int)
    # wind applied at each state as array of size Nx2
    world['wind'] = np.zeros((world['states'], 2))
    if not wind is None:
        world['wind'][wind[:, 0].astype(int)] = wind[:, 1:]
    # invalid states and transitions as lists
    world['invalid_states'] = invalid_states
    world['invalid_transitions'] = invalid_transitions
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

if __name__ == '__main__':
    height, width = 5, 5
    goal = 4
    reward = 1
    columns = np.array([0, 0, 0, 1, 1])
    gridworld = make_gridworld(height, width, terminals=[goal], rewards=np.array([[goal, reward]]))
    open_field = make_open_field(height, width, goal, reward)
    windy_gridworld = make_windy_gridworld(height, width, columns, goal, reward)
