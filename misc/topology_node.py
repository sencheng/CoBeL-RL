
from .cog_arrow import CogArrow

### This class defines a single node of the topology graph.
class TopologyNode():
    def __init__(self,index,x,y):
        # the node's global index
        self.index=index
        
        # the node's global position
        self.x=x
        self.y=y
        
        # is this node the requested goal node?
        self.goalNode=False
        
        # the clique of the node's neighboring nodes
        self.neighbors=[]
        
        # an indicator arrow that points in the direction of the most probable next neighbor (as planned by the RL system)
        self.qIndicator=CogArrow()

        # if not otherwise defined or inhibited, each node is also a starting node
        self.startNode=False
        
        # this reward bias is assigned to the node as standard (0.0), and can be changed dynamically to reflect environmental reconfigurations of rewards
        self.nodeRewardBias=0.0
        
