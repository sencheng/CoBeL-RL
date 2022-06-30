# basic imports
import numpy as np
import os
# framework imports
import blender_frontend
# blender imports
import bge


class CustomBlenderFrontend(blender_frontend.BlenderFrontend):
    '''
    Custom Blender frontend.
    
    | **Args**
    | control_buffer_size:          The buffer size of the control connection when receiving commands from the framework.
    '''
    
    def __init__(self, control_buffer_size=500):
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
        
    def set_render_state(self, data):
        '''
        This function sets the render state of a given barrier object. If a barrier
        object is being rendered and blocks edges and/or vertices of the topology
        graph, then the topology graph must be adapted accordingly.
        
        | **Args**
        | barrier_ID:                   The barrier object's identifier.
        | render_state:                 The barrier object's render state (True/False).
        '''
        # split data string
        barrier_ID, render_state = data
        # retrieve barrier object and the edge associated with it from scene
        barrier_object = self.scene.objects[barrier_ID]
        graphEdge = self.scene.objects[barrier_object['obstructedEdge']]
        #If the barrier is rendered, the corresponding edge is removed from the graph and vice versa
        if 'True' in render_state:
            barrier_object.setVisible(True)
            graphEdge['graphEdge'] = False
        elif 'False' in render_state:
            barrier_object.setVisible(False)
            graphEdge['graphEdge'] = True
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
    
    def set_rotation(self, data):
        '''
        This function sets the rotation of a given barrier object.
        
        | **Args**
        | barrier_ID:                   The barrier object's identifier.
        | rotation:                     The barrier object's new orientation.
        '''
        # split data string
        barrier_ID, rotation = data
        # recover rotation in radians
        rotation = np.radians(float(rotation))
        # retrieve barrier object from scene
        barrier_object = self.scene.objects[barrier_ID]
        barrier_object.applyRotation([0., 0., rotation])
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
    
    def set_texture(self, data):
        '''
        This function sets the texture of a given barrier object.
        
        | **Args**
        | barrier_ID:                   The barrier object's identifier.
        | texture:                      The texture to be applied to the object.
        '''
        # split data string
        barrier_ID, texture = data
        # retrieve barrier object from scene
        barrier_object = self.scene.objects[barrier_ID]
        #Get the ID of the internal texture
        mesh = barrier_object.meshes[0]
        texture_name = mesh.getTextureName(0)
        material_ID = bge.texture.materialID(barrier_object, texture_name)
        # apply the texture unless the filepath is invalid
        try:
            # create new texture object
            object_texture = bge.texture.Texture(barrier_object, material_ID)
            #Create a new source pointing to the image file
            url = bge.logic.expandPath(texture)
            # if the file does not exist raise error
            if not(os.path.exists(url)):
                print('Error. File not found.')
                raise FileNotFoundError('File not found')
            new_source = bge.texture.ImageFFmpeg(url)
            # store dynamic texture
            self.dynamic_textures[barrier_ID] = object_texture
            self.dynamic_textures[barrier_ID].source = new_source
            self.dynamic_textures[barrier_ID].refresh(True)
            # add filepath of the new texture to object properties
            barrier_object['dynamicTexture'] = texture
            # send control data
            self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        # if an error was raised due to an invalid filepath send 'NAK' to the controll port
        except FileNotFoundError:
            self.controller['controlConnection'].send('FileNotFoundError.'.encode('utf-8'))
    
    def set_spotlight(self, data):
        '''
        This function sets the render state of a given spotlight object.
        Spotlights light up the area around the given topology graph node.
        
        | **Args**
        | spotlight_ID:                 The spotlight object's identifier.
        | render_state:                 The spotlight object's render state (True/False).
        '''
        # split data string
        spotlight_ID, render_state = data
        # retrieve spotlight object from scene
        spotlight_object = self.scene.objects[spotlight_ID]
        # toggle visibility of given spotlight object
        if 'True' in render_state:
            # enable spotlight by setting illumination to 100%
            spotlight_object.energy = 1.
        elif 'False' in render_state:
            # disable spotlight by setting illumination to 0%
            spotlight_object.energy = 0.
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
    
    def get_barrier_info(self, data):
        '''
        This function sends back render state, rotation and texture information of a given barrier object.
        
        | **Args**
        | barrier_ID:                   The barrier object's identifier.
        '''
        # split data string
        barrier_ID = data
        # retrieve object as well as its render state and orientation
        barrier_object = self.scene.objects[barrier_ID]
        render_state = barrier_object.visible
        orientation = barrier_object.localOrientation
        # try to retrieve the dynamic texture
        texture = 'DEFAULT'
        if 'dynamicTexture' in barrier_object:
            texture = barrier_object
        # convert rotation matrix to string
        rotation = ''
        for i in orientation:
            for j in i:
                rotation += '%f;' % j
            rotation = rotation[:-1]
            rotation += '|'
        rotation = rotation[:-1]
        # prepare response string
        response = '%s,%s,%s' % (render_state, rotation, texture)
        # send control data
        self.controller['controlConnection'].send(response.encode('utf-8'))
    
    def get_barrier_IDs(self):
        '''
        This function sends back a list with all barrier object identities.
        '''
        sendStr = ''
        # collect barrier objects
        for scene_object in self.scene.objects:
            if 'barrier' in scene_object.name:
                sendStr += scene_object.name + ','
        # clean up the data string
        if sendStr.endswith(','):
            sendStr = sendStr[:-1]
        # send control data
        self.controller['controlConnection'].send(sendStr.encode('utf-8'))
        
    def getManuallyDefinedTopologyNodes(self):
        '''
        This function sends back the manually defined topology nodes.
        '''
        # retrieve the manually defined topology nodes from scene
        manuallyDefinedTopologyNodes = []
        for obj in self.scene.objects:
            if 'graphNode' in obj.name:
                if obj['graphNode'] == True:
                    manuallyDefinedTopologyNodes += [obj]
        # send the number of nodes
        print('found %d manually defined topology nodes' % len(manuallyDefinedTopologyNodes))
        retStr = '%.d' % len(manuallyDefinedTopologyNodes)
        # send control data
        self.controller['controlConnection'].send(retStr.encode('utf-8'))
        # wait for the framework's answer
        self.controller['controlConnection'].recv(1000)
        # loop over nodes
        for obj in manuallyDefinedTopologyNodes:                
            # retrieve node information
            name = obj.name
            xPos = obj.worldPosition.x
            yPos = obj.worldPosition.y
            nodeType = 'standardNode'
            if 'startNode' in obj:
                nodeType = 'startNode'
            if 'goalNode' in obj:
                nodeType = 'goalNode'
            # send node information
            retStr = '%s,%f,%f,%s' % (name, xPos, yPos, nodeType)
            # send control data
            self.controller['controlConnection'].send(retStr.encode('utf-8'))
            # wait for the framework's answer
            self.controller['controlConnection'].recv(1000)        
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def getManuallyDefinedTopologyEdges(self):
        '''
        This function sends back the manually defined topology edges.
        '''
        # retrieve the manually defined topology edges from scene
        manuallyDefinedTopologyEdges = []
        for obj in self.scene.objects:
            if 'graphEdge' in obj.name:
                if obj['graphEdge'] == True:
                    manuallyDefinedTopologyEdges += [obj]
        # send the number of edges
        print('found %d manually defined topology edges' % len(manuallyDefinedTopologyEdges))
        retStr = '%.d' % len(manuallyDefinedTopologyEdges)
        # send control data
        self.controller['controlConnection'].send(retStr.encode('utf-8'))
        # wait for the framework's answer
        self.controller['controlConnection'].recv(1000)
        # loop over edges
        for obj in manuallyDefinedTopologyEdges:
            # retrieve edge information
            name = obj.name
            first = obj['first']
            second = obj['second']
            # send edge information
            retStr = '%s,%d,%d' % (name, first, second)
            # send control data
            self.controller['controlConnection'].send(retStr.encode('utf-8'))
            # wait for the framework's answer
            self.controller['controlConnection'].recv(1000)
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
    

BF = CustomBlenderFrontend()
BF.main_loop()