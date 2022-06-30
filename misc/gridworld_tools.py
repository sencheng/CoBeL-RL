# basic imports
import numpy as np


def makeGridworld(height, width, terminals=[], rewards=None, goals=[], startingStates=[], invalidStates=[], invalidTransitions=[], wind=None, deterministic=True):
    '''
    This function builds a gridworld according to the given parameters.
    
    | **Args**
    | height:                       The gridworld's height.
    | width:                        The gridworld's width.
    | terminals:                    The gridworld's terminal states as a list.
    | rewards:                      The gridworld's state rewards as an array where the first column contains the state indeces and the second column the state rewards. 
    | goals:                        The gridworld's goal states as a list (Used for visualization).
    | startingStates:               Possible starting states as a list.
    | invalidStates:                The gridworld's unreachable states as list.
    | invalidTransitions:           The gridworld's invalid transitions as a list of 2-tuples.
    | wind:                         The wind applied to the gridworld's states where the first column contains the state indeces and the second and thirds column the wind applied to height and width coordinates.
    | deterministic:                If true, state transition with the highest probability are chosen.
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
    world['startingStates'] = list(set([i for i in range(world['states'])]) - set(terminals))
    if len(startingStates) > 0:
        world['startingStates'] = startingStates
    world['startingStates'] = np.array(world['startingStates']).astype(int)
    # wind applied at each state as array of size Nx2
    world['wind'] = np.zeros((world['states'], 2))
    if not wind is None:
        world['wind'][wind[:, 0].astype(int)] = wind[:, 1:]
    # invalid states and transitions as lists
    world['invalidStates'] = invalidStates
    world['invalidTransitions'] = invalidTransitions
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
            if nextState in world['invalidStates'] or (state, nextState) in world['invalidTransitions']:
                nextState = state
            world['sas'][state][action][nextState] = 1
            
    world['deterministic'] = deterministic
    
    return world
    
def makeOpenField(height, width, goalState=0, reward=1):
    '''
    This function builds an open field gridworld with one terminal goal state.
    
    | **Args**
    | height:                       The gridworld's height.
    | width:                        The gridworld's width.
    | goalState:                    The gridworld's goal state.
    | reward:                       The reward received upon reaching the gridworld's goal state.
    '''
    return makeGridworld(height, width, terminals=[goalState], rewards=np.array([[goalState, reward]]), goals=[goalState])

def makeEmptyField(height, width):
    '''
    This function builds an empty open field gridworld.
    
    | **Args**
    | height:                       The gridworld's height.
    | width:                        The gridworld's width.
    '''
    return makeGridworld(height, width)

def makeWindyGridworld(height, width, columns, goalState=0, reward=1, direction='up'):
    '''
    This function builds a windy gridworld with one terminal goal state.
    
    | **Args**
    | height:                       The gridworld's height.
    | width:                        The gridworld's width.
    | columns:                      Wind strengths for the different columns.
    | goalState:                    The gridworld's goal state.
    | reward:                       The reward received upon reaching the gridworld's goal state.
    | direction:                    The wind's direction (up, down).
    '''
    directions = {'up': 1, 'down': -1}
    wind = np.zeros((height * width, 3))
    for i in range(width):
        for j in range(height):
            state = int(j * width + i)
            wind[state] = np.array([state, columns[i] * directions[direction], 0])
    return makeGridworld(height, width, terminals=[goalState], rewards=np.array([[goalState, reward]]), goals=[goalState], wind=wind)

if __name__ == '__main__':
    height, width = 5, 5
    goal = 4
    reward = 1
    columns = np.array([0, 0, 0, 1, 1])
    gridworld = makeGridworld(height, width, terminals=[goal], rewards=np.array([[goal, reward]]))
    openField = makeOpenField(height, width, goal, reward)
    windyGridworld = makeWindyGridworld(height, width, columns, goal, reward)