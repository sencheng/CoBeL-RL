# basic imports
import os
import math
# framework imports
import blender_frontend
# blender imports
import bge


class DynamicBlenderFrontend(blender_frontend.BlenderFrontend):
    
    def __init__(self, control_buffer_size=1024):
        '''
        Custom Blender frontend.
        
        Parameters
        ----------
        control_buffer_size :               The buffer size of the control connection when receiving commands from the framework.
        
        Returns
        ----------
        None
        '''
        super().__init__(control_buffer_size=control_buffer_size)
        # stores dynamic textures
        self.dynamic_textures = {}
        # add custom functions accessible during the main loop
        self.functions['set_render_state'] = self.set_render_state
        self.functions['set_rotation'] = self.set_rotation
        self.functions['set_texture'] = self.set_texture
        self.functions['set_spotlight'] = self.set_spotlight
        self.functions['get_barrier_info'] = self.get_barrier_info
        self.functions['get_barrier_IDs'] = self.get_barrier_IDs
        
    def set_render_state(self, barrier_ID: str, render_state: bool):
        '''
        This function sets the render state of a given barrier object. If a barrier
        object is being rendered and blocks edges and/or vertices of the topology
        graph, then the topology graph must be adapted accordingly.
        
        Parameters
        ----------
        barrier_ID :                        The barrier object's identifier.
        render_state :                      The barrier object's render state (True/False).
        
        Returns
        ----------
        None
        '''
        # retrieve barrier object and the edge associated with it from scene
        barrier_object = self.scene.objects[barrier_ID]
        graph_edge = self.scene.objects[barrier_object['obstructedEdge']]
        #If the barrier is rendered, the corresponding edge is removed from the graph and vice versa
        barrier_object.setVisible(render_state)
        graph_edge['graphEdge'] = not render_state
    
    def set_rotation(self, barrier_ID: str, rotation: float):
        '''
        This function sets the rotation of a given barrier object.
        
        Parameters
        ----------
        barrier_ID :                        The barrier object's identifier.
        rotation :                          The barrier object's new orientation.
        
        Returns
        ----------
        None
        '''
        # retrieve barrier object from scene and apply rotation
        barrier_object = self.scene.objects[barrier_ID]
        barrier_object.applyRotation([0., 0., rotation])
    
    def set_texture(self, barrier_ID: str, texture: str):
        '''
        This function sets the texture of a given barrier object.
        
        Parameters
        ----------
        barrier_ID :                        The barrier object's identifier.
        texture :                           The texture to be applied to the object.
        
        Returns
        ----------
        None
        '''
        # retrieve barrier object from scene
        barrier_object = self.scene.objects[barrier_ID]
        #Get the ID of the internal texture
        mesh = barrier_object.meshes[0]
        texture_name = mesh.getTextureName(0)
        material_ID = bge.texture.materialID(barrier_object, texture_name)
        # create new texture object
        object_texture = bge.texture.Texture(barrier_object, material_ID)
        #Create a new source pointing to the image file
        url = bge.logic.expandPath(texture)
        # if the file does not exist raise error
        if os.path.exists(url):
            new_source = bge.texture.ImageFFmpeg(url)
            # store dynamic texture
            self.dynamic_textures[barrier_ID] = object_texture
            self.dynamic_textures[barrier_ID].source = new_source
            self.dynamic_textures[barrier_ID].refresh(True)
            # add filepath of the new texture to object properties
            barrier_object['dynamicTexture'] = texture
    
    def set_spotlight(self, spotlight_ID: str, render_state: bool):
        '''
        This function sets the render state of a given spotlight object.
        Spotlights light up the area around the given topology graph node.
        
        Parameters
        ----------
        spotlight_ID :                      The spotlight object's identifier.
        render_state :                      The spotlight object's render state (True/False).
        
        Returns
        ----------
        None
        '''
        # retrieve spotlight object from scene
        spotlight_object = self.scene.objects[spotlight_ID]
        # set visibility of the given spotlight object
        spotlight_object.energy = float(render_state)
    
    def get_barrier_info(self, barrier_ID: str):
        '''
        This function sends back render state, rotation and texture information of a given barrier object.
        
        Parameters
        ----------
        barrier_ID :                        The barrier object's identifier.
        
        Returns
        ----------
        None
        '''
        # retrieve object as well as its render state and orientation
        barrier_object = self.scene.objects[barrier_ID]
        # try to retrieve the dynamic texture
        texture = 'DEFAULT'
        if 'dynamicTexture' in barrier_object:
            texture = barrier_object['dynamicTexture']
        # convert rotation matrix to string
        rotation = barrier_object.localOrientation.to_euler('XYZ')
        rotation = math.degrees(rotation[2])
        # send data
        self.send_data(self.controller['data_connection'], [barrier_object.visible, rotation, texture])
    
    def get_barrier_IDs(self):
        '''
        This function sends back a list with all barrier object identities.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # collect barrier objects
        barrier_IDs = []
        for scene_object in self.scene.objects:
            if 'barrier' in scene_object.name:
                barrier_IDs.append(scene_object.name)
        # send data
        self.send_data(self.controller['data_connection'], barrier_IDs)
        
    def get_manually_defined_topology_nodes(self):
        '''
        This function sends back the manually defined topology nodes.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve nodes
        nodes = []
        for obj in self.scene.objects:
            if 'graphNode' in obj.name:
                if obj['graphNode'] == True:
                    node_type = 'standardNode'
                    if 'startNode' in obj:
                        node_type = 'startNode'
                    elif 'goalNode' in obj:
                        node_type = 'goalNode'
                    nodes.append([obj.name, obj.worldPosition.x, obj.worldPosition.y, node_type])
        # send data
        self.send_data(self.controller['data_connection'], nodes)
        
    def get_manually_defined_topology_edges(self):
        '''
        This function sends back the manually defined topology edges.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # retrieve edges
        edges = []
        for obj in self.scene.objects:
            if 'graphEdge' in obj.name:
                if obj['graphEdge'] == True:
                    edges.append([obj.name, obj['first'], obj['second']])
        # send data
        self.send_data(self.controller['data_connection'], edges)
