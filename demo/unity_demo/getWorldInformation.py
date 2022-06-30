# basic imports
import os
import pickle
import pyqtgraph as qg
# framework imports
from cobel.frontends.frontends_unity import FrontendUnityInterface
from cobel.spatial_representations.topology_graphs.four_connected_graph_rotation import Four_Connected_Graph_Rotation
from cobel.observations.image_observations import ImageObservationUnity

# shall the system provide visual output while performing the experiments?
# NOTE: do NOT use visualOutput=True in parallel experiments, visualOutput=True should only be used in explicit calls to 'singleRun'! 
visualOutput = True


if __name__ == "__main__":
    mainWindow = None
    # if visual output is required, activate an output window
    mainWindow = qg.GraphicsWindow(title="workingTitle_Framework")
    # determine local framework path
    path = os.path.abspath(__file__).split('cobel')[0] + '/cobel/'
    # a dictionary that contains all employed modules
    modules = {}
    modules['world'] = FrontendUnityInterface('TMaze')
    modules['observation'] = ImageObservationUnity(modules['world'], mainWindow, visualOutput, False)
    modules['spatial_representation'] = Four_Connected_Graph_Rotation(modules, {'startNodes':[3], 'goalNodes':[0], 'start_ori': 90, 'cliqueSize':4}, step_size=1.)
    modules['spatial_representation'].set_visual_debugging(True, mainWindow)
    # delete previous directory contents
    os.system('rm -rf ' + path + 'demo/unity_demo/worldInfo')
    # re-create the worldInfo directory
    os.makedirs(path + 'demo/unity_demo/worldInfo', exist_ok=True)
    # collect world info
    worldInfo = {}
    worldInfo['world_limits'] = modules['world'].world_limits
    worldInfo['walls_limits'] = modules['world'].walls_limits
    worldInfo['perimeter_nodes'] = modules['world'].perimeter_nodes
    # collect observations
    for ni in range(len(modules['spatial_representation'].nodes)):
        node = modules['spatial_representation'].nodes[ni]
        # only for valid nodes, the 'NoneNode' is not considered here
        if node.index != -1:
            for orientation in [0, 90, 180, 270, -90, -180, -270]:
                # propel the simulation
                modules['spatial_representation'].nextNode = node.index
                modules['world'].step_simulation_without_physics(node.x, node.y, orientation)
                # the observation is plainly the robot's camera image data
                worldInfo[str([node.index, orientation])] = modules['world'].envData['imageData']
    pickle.dump(worldInfo, open(path + 'demo/unity_demo/worldInfo/TMaze_Infos.pkl', 'wb'))
    # stop Unity
    modules['world'].stopUnity()
    # and also stop visualization
    if visualOutput:
        mainWindow.close()