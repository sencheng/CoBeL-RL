# basic imports
import numpy as np
import os
import pyqtgraph as qg
import cv2
# framework imports
from cobel.frontends.frontends_blender import FrontendBlenderInterface
from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation
from cobel.observations.image_observations import ImageObservationBaseline


if __name__ == '__main__':
    mainWindow = None
    # if visual output is required, activate an output window
    mainWindow = qg.GraphicsWindow(title="workingTitle_Framework")
    
    # determine local framework path
    path = os.path.abspath(__file__).split('cobel')[0] + '/cobel/'
    # determine demo scene path
    demo_scene = path + 'environments/environments_blender/simple_grid_graph_maze.blend'
    
    # a dictionary that contains all employed modules
    modules=dict()
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], mainWindow, True)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules,{'startNodes':[0], 'goalNodes':[15], 'cliqueSize':4})
    modules['spatial_representation'].set_visual_debugging(True, mainWindow)
    
    # delete previous directory contents
    os.system('rm -rf ' + path + 'demo/offline_demo/worldInfo')
    # re-create the worldInfo directory
    os.makedirs(path + 'demo/offline_demo/worldInfo', exist_ok=True)
   
    safeZoneDimensions = np.array([modules['world'].minX, modules['world'].minY, modules['world'].minZ,
                                   modules['world'].maxX, modules['world'].maxY, modules['world'].maxZ])
    np.save(path + 'demo/offline_demo/worldInfo/safeZoneDimensions.npy', safeZoneDimensions)
    np.save(path + 'demo/offline_demo/worldInfo/safeZonePolygon.npy', modules['world'].safeZonePolygon)
    np.save(path + 'demo/offline_demo/worldInfo/safeZoneVertices.npy', modules['world'].safeZoneVertices)
    np.save(path + 'demo/offline_demo/worldInfo/safeZoneSegments.npy', modules['world'].safeZoneSegments)
   
    # store environment information
    nodes = np.array(modules['world'].getManuallyDefinedTopologyNodes())
    nodes = nodes[nodes[:, 0].argsort()]
    edges = np.array(modules['world'].getManuallyDefinedTopologyEdges())
    edges = edges[edges[:, 0].argsort()]
    np.save(path + 'demo/offline_demo/worldInfo/topologyNodes.npy', nodes)
    np.save(path + 'demo/offline_demo/worldInfo/topologyEdges.npy', edges)
   
   
   
    # store referenceImages sampled images from contexts A and B
    
    # Note: context A is conditioning and ROF (both red light), context b is extinction (white light). The light colors can be changed on demand.

    # prepare context A
    # switch on white light
    
    imageDims = (30, 1)
    referenceImages = []
    
    for ni in range(len(modules['spatial_representation'].nodes)):
        node = modules['spatial_representation'].nodes[ni]
        # only for valid nodes, the 'NoneNode' is not considered here
        if node.index != -1:
            # propel the simulation
            modules['spatial_representation'].nextNode = node.index # required by the WORLD_ABAFromImagesInterface            
            modules['world'].step_simulation_without_physics(node.x, node.y, 90.0)
            # the observation is plainly the robot's camera image data
            observation = modules['world'].envData['imageData']
            # for now, cut out a single line from the center of the image (w.r.t. height) and use this as an observation in order
            # to save computational resources
            #observation=observation[29:30,:,:]
            # scale the one-line image to further reduce computational demands
            observation = cv2.resize(observation, dsize=(imageDims))
            #observation=np.flip(observation,0)
            #cv2.imshow('Test',observation)
            #cv2.waitKey(0)
            referenceImages += [observation]
    
    images = np.array(referenceImages)
    np.save(path + 'demo/offline_demo/worldInfo/images.npy', images)
    modules['world'].stopBlender()
    # and also stop visualization
    mainWindow.close()