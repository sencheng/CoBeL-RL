import numpy as np
import gym

class PoseObservation() : 
    
    def __init__(self, topology, gui_parent) :
                
        self.topology        = topology
        self.gui_parent      = gui_parent
        
    def add_topology_graph(self, topology_graph) : 
        self.topology = topology_graph
    
    def update(self) : 
        '''
        Updates the observation. For observing a vector to the goal, this will
        compute the vector between the current node on the topology graph and
        the goal node.

        Returns
        -------
        None.

        '''
        current_node  = self.topology.current_node

        current_x = self.topology.nodes[current_node].x
        current_y = self.topology.nodes[current_node].y
        
        head_direction    = self.topology.head_direction
        hd_rad = np.deg2rad(head_direction)
        
        self.observation = np.array([current_x, current_y, hd_rad])
    
    def set_observation_state(self, state) :
        self._observe = state
    
    def getObservationSpace(self):
        
        '''
        This function returns the observation space for the given observation class.
        '''
        return gym.spaces.Box (low=0.0, high=10.0, shape=(3,))