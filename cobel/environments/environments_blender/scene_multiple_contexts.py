# blender imports
import bge
# framework imports
import blender_frontend


class MultipleContextsBF(blender_frontend.BlenderFrontend):
    '''
    Custom Blender frontend.
    
    | **Args**
    | control_buffer_size:          The buffer size of the control connection when receiving commands from the framework.
    '''
    
    def __init__(self, control_buffer_size=1000):
        super().__init__(control_buffer_size=control_buffer_size)
        # prepare dynamic textures
        box = self.scene.objects['Skybox']
        for i in range(4):
            box[str(i) + ':0'] = bge.texture.Texture(box, i, 0)
        # add custom functions accessible during the main loop
        self.functions['set_wall_textures'] = self.set_wall_textures
        
    def set_wall_textures(self, data):
        '''
        This function updates the wall textures of the box.
        
        | **Args**
        | left_wall_texture:            The texture that will be applied to the left wall.
        | front_wall_texture:           The texture that will be applied to the front wall.
        | right_wall_texture:           The texture that will be applied to the right wall.
        | back_wall_texture:            The texture that will be applied to the back wall.
        '''
        # recover textures
        textures = data
        # retrieve box object from scene
        box = self.scene.objects['Skybox']
        # update texture sources
        for t, texture in enumerate(textures):
            box[str(t) + ':0'].source = bge.texture.ImageFFmpeg(texture)
            box[str(t) + ':0'].refresh(True)
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        

BF = MultipleContextsBF()
BF.main_loop()