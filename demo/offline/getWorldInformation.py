# basic imports
import os
import cv2
import numpy as np
import pyqtgraph as qg
# framework imports
from cobel.frontends.frontends_blender import FrontendBlenderInterface
from cobel.observations.image_observations import ImageObservationBaseline
from cobel.spatial_representations.topology_graphs.manual_topology_graph_no_rotation import ManualTopologyGraphNoRotation


if __name__ == '__main__':
    mainWindow = None
    # if visual output is required, activate an output window
    mainWindow = qg.GraphicsLayoutWidget(title="workingTitle_Framework")
    mainWindow.show()
    
    # determine demo scene path
    demo_scene = '../../environments/environments_blender/simple_grid_graph_maze.blend'
    
    # a dictionary that contains all employed modules
    modules=dict()
    modules['world'] = FrontendBlenderInterface(demo_scene)
    modules['observation'] = ImageObservationBaseline(modules['world'], mainWindow, True)
    modules['spatial_representation'] = ManualTopologyGraphNoRotation(modules,{'start_nodes':[0], 'goal_nodes':[15], 'clique_size':4})
    modules['spatial_representation'].set_visual_debugging(True, mainWindow)
    
    # re-create the worldInfo directory
    os.makedirs('worldInfo', exist_ok=True)
   
    safeZoneDimensions = np.array([modules['world'].min_x, modules['world'].min_y, modules['world'].min_z,
                                   modules['world'].max_x, modules['world'].max_y, modules['world'].max_z])
    np.save('worldInfo/safeZoneDimensions.npy', safeZoneDimensions)
    np.save('worldInfo/safeZonePolygon.npy', modules['world'].safe_zone_polygon)
    np.save('worldInfo/safeZoneVertices.npy', modules['world'].safe_zone_vertices)
    np.save('worldInfo/safeZoneSegments.npy', modules['world'].safe_zone_segments)
   
    # store environment information
    nodes = np.array(modules['world'].get_manually_defined_topology_nodes())
    nodes = nodes[nodes[:, 0].argsort()]
    edges = np.array(modules['world'].get_manually_defined_topology_edges())
    edges = edges[edges[:, 0].argsort()]
    np.save('worldInfo/topologyNodes.npy', nodes)
    np.save('worldInfo/topologyEdges.npy', edges)
   
   
   
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
            modules['spatial_representation'].next_node = node.index # required by the WORLD_ABAFromImagesInterface            
            modules['world'].step_simulation_without_physics(node.x, node.y, 90.0)
            # the observation is plainly the robot's camera image data
            observation = modules['world'].env_data['image']
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
    np.save('worldInfo/images.npy', images)
    modules['world'].stop_blender()
    # and also stop visualization
    mainWindow.close()
