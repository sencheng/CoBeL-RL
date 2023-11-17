# basic imports
import numpy as np
import pyqtgraph as qg
from PyQt5.QtCore import QRectF
# open-cv
import cv2
# OpenAI Gym
import gymnasium as gym

 
class ImageObservationBaseline():
    
    def __init__(self, world, gui_parent: None | qg.GraphicsLayoutWidget = None, with_GUI: bool = True, image_dims: tuple = (30, 1)):
        '''
        This module computes an observation based on the current camera image acquired by the robot/agent.
        Raw image data is processed to line image observations.
        
        Parameters
        ----------
        world :                             The world module.
        with_GUI :                          If true, observations and policy will be visualized.
        gui_parent :                        The main window used for visualization.
        image_dims :                        The dimensions the observation will be resized to.
        
        Returns
        ----------
        None
        '''
        # store the world module reference
        self.world_module = world
        self.topology_module = None
        self.with_GUI = with_GUI
        self.gui_parent = gui_parent
        self.observation = None
        # generate a visual display of the observation
        if self.with_GUI:
                # add the graph plot to the GUI widget
                self.plot = self.gui_parent.addPlot(title='Camera Image Observation')
                # set extension of the plot, lock aspect ratio
                self.plot.setAspectLocked()
                # add the camera image plot item
                self.camera_image = qg.ImageItem()
                self.plot.addItem(self.camera_image)
                # add the observation plot item
                self.observation_image = qg.ImageItem()
                self.plot.addItem(self.observation_image)
        self.image_dims = image_dims
        # STD for Gaussian noise
        self.noise_gaussian = 0.
        # probability for a pixel to turn hot/cold
        self.noise_salt_pepper = 0.
        # the image format
        self.format = 'bgr'
    
    def update(self):
        '''
        This function processes the raw image data and updates the current observation.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # the observation is plainly the robot's camera image data
        observation = self.world_module.env_data['image']
        # display the observation camera image
        if self.with_GUI:
            image_data = observation
            self.camera_image.setOpts(axisOrder='row-major')
            if self.format == 'bgr':
                image_data = image_data[:,:,::-1]
            self.camera_image.setImage(image_data)
            image_scale = 1.0
            self.camera_image.setRect(QRectF(0.0, 0.0, image_scale, image_data.shape[0]/image_data.shape[1]*image_scale))
        # scale the one-line image to further reduce computational demands
        observation = cv2.resize(observation, dsize=self.image_dims)
        observation.astype('float32')
        observation = observation/255.0
        # apply Gaussian noise
        if self.noise_gaussian > 0.:
            observation = np.clip(np.random.normal(observation, self.noise_gaussian), a_min=0., a_max=1.)
        # apply Salt and Pepper noise
        if self.noise_salt_pepper > 0:
            observation = observation.flatten()
            idx = np.random.rand(observation.shape[0])
            observation[idx < self.noise_salt_pepper/2] = 0.
            observation[idx > self.noise_salt_pepper/2] = 1.
            observation = observation.reshape(self.image_dims)
        # display the observation camera image reduced to one line
        if self.with_GUI:
            image_data = observation
            self.observation_image.setOpts(axisOrder='row-major')
            if self.format == 'bgr':
                image_data = image_data[:,:,::-1]
            self.observation_image.setImage(image_data)
            image_scale = 1.0
            self.observation_image.setRect(QRectF(0.0, -0.1, image_scale, image_data.shape[0]/image_data.shape[1]*image_scale))
        self.observation = observation
        
    def get_observation_space(self):
        '''
        This function returns the observation space for the given observation class.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation_space :                 The observation space.
        '''
        '''
        This function returns the observation space for the given observation class.
        '''
        # currently, use a one-line 'image' to save computational resources
        return gym.spaces.Box(low=0.0, high=1.0, shape=(self.image_dims[1], self.image_dims[0], 3))
    
    
class ImageObservationUnity():
    
    def __init__(self, world, gui_parent: None | qg.GraphicsLayoutWidget = None, with_GUI: bool = True,
                 use_gray_scale: bool = False, image_dims: tuple = (84, 84, 3)):
        '''
        This module computes an observation based on the current camera image acquired by the robot/agent.
        Raw image data is processed to line image observations.
        
        Parameters
        ----------
        world :                             The world module.
        with_GUI :                          If true, observations and policy will be visualized.
        gui_parent :                        The main window used for visualization.
        use_gray_scale :                    If true, RGB images will be converted into gray scale images.
        image_dims :                        The dimensions the observation will be resized to.
        
        Returns
        ----------
        None
        '''
        # store the world module reference
        self.world_module = world
        self.gui_parent = gui_parent
        self.topology_module = None
        self.with_GUI = with_GUI
        self.observation = None
        self.use_gray_scale = use_gray_scale
        # generate a visual display of the observation
        if self.with_GUI:
            # add the graph plot to the GUI widget
            self.plot = self.gui_parent.addPlot(title='Camera Image Observation')
            # set extension of the plot, lock aspect ratio
            self.plot.setAspectLocked()
            # add the camera image plot item
            self.camera_image = qg.ImageItem()
            self.plot.addItem(self.camera_image)
            # add the observation plot item
            self.observation_image = qg.ImageItem()
            self.plot.addItem(self.observation_image)

        self.image_dims = image_dims
        if self.use_gray_scale:
            self.image_dims = (self.image_dims[0], self.image_dims[1], 1)

    def update(self):
        '''
        This function processes the raw image data and updates the current observation.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        None
        '''
        # the robot's/agent's pose has changed, get the new observation by evaluating
        # information from the world module:
        # the observation is plainly the robot's camera image data
        observation = self.world_module.env_data['image']
        if self.use_gray_scale:
            observation = self.to_gray_scale_image(observation)
        observation = cv2.resize(observation, dsize=self.image_dims[:2])
        # display the observation camera image
        if self.with_GUI:
            image_data = observation
            self.camera_image.setOpts(axisOrder='row-major')
            # mirror the image
            self.camera_image.setImage(image_data[::-1])
            self.camera_image.setRect(QRectF(0.0, 0.0, self.image_dims[0], self.image_dims[1]))
            self.camera_image.setLevels([0.0, 0.9])
        self.observation = observation
    
    def get_observation_space(self):
        '''
        This function returns the observation space for the given observation class.
        
        Parameters
        ----------
        None
        
        Returns
        ----------
        observation_space :                 The observation space.
        '''
        '''
        This function returns the observation space for the given observation class.
        '''
        return gym.spaces.Box(low=0.0, high=1.0,shape=(self.image_dims[1], self.image_dims[0], self.image_dims[2]))

    def to_gray_scale_image(self, image_array: np.ndarray) -> np.ndarray:
        '''
        This function converts a 3D image array to a 2D grayscale image array.
        
        Parameters
        ----------
        image_array :                       The image that will be converted to grayscale.
        
        Returns
        ----------
        gray_scale_image :                  The grayscaled image.
        '''
        assert len(image_array.shape) == 3, 'provided image does not match the expected shape'
        # convert to greyscale
        gray_scale_image = np.sum(image_array[:, :, :3], axis=2) / image_array.shape[2]
        # adjust contrast
        contrast = 1
        gray_scale_image = contrast * (gray_scale_image - 0.5) + 0.5
        # add the additional channel number of 1
        gray_scale_image = np.reshape(gray_scale_image, (gray_scale_image.shape[0], gray_scale_image.shape[1], 1))

        return gray_scale_image
