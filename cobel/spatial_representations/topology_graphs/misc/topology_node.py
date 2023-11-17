# framework imports
from .cog_arrow import CogArrow


class TopologyNode():
    
    def __init__(self, index: int, x: float, y: float):
        '''
        This class defines a single node of the topology graph.
        
        Parameters
        ----------
        index :                             The node's global index.\n
        x :                                 The node's x position.\n
        y :                                 The node's y position.\n
        
        Returns
        ----------
        None
        '''
        # the node's global index
        self.index = index
        # the node's global position
        self.x, self.y = x, y
        # is this node the requested goal node?
        self.goal_node = False
        # the clique of the node's neighboring nodes
        self.neighbors = []
        # an indicator arrow that points in the direction of the most probable next neighbor (as planned by the RL system)
        self.q_indicator = CogArrow()
        # if not otherwise defined or inhibited, each node is also a starting node
        self.start_node = False
        # this reward bias is assigned to the node as standard (0.0)
        self.node_reward_bias = 0.
