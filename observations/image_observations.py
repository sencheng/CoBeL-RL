# basic imports
import numpy as np
import pyqtgraph as qg
from PyQt5.QtCore import QRectF
# open-cv
import cv2
# OpenAI Gym
import gym

 
class ImageObservationBaseline():
    '''
    This module computes an observation based on the current camera image acquired by the robot/agent.
    Raw image data is processed to line image observations.
    
    | **Args**
    | world:                        The world module.
    | guiParent:                    The main window for visualization.
    | visualOutput:                 Does the module provide visual output?
    | imageDims:                    The dimensions the observation will be resized to.
    '''
    
    def __init__(self, world, guiParent, visualOutput=True, imageDims=(30, 1)):
        # store the world module reference
        self.worldModule = world
        self.topologyModule = None
        self.visualOutput = visualOutput
        self.observation = None
        # generate a visual display of the observation
        if self.visualOutput:
                # add the graph plot to the GUI widget
                self.plot = guiParent.addPlot(title='Camera image observation')
                # set extension of the plot, lock aspect ratio
                self.plot.setAspectLocked()
                # add the camera image plot item
                self.cameraImage = qg.ImageItem()
                self.plot.addItem(self.cameraImage)
                # add the observation plot item
                self.observationImage = qg.ImageItem()
                self.plot.addItem(self.observationImage)
        self.imageDims = imageDims
    
    def update(self):
        '''
        This function processes the raw image data and updates the current observation.
        '''
        # the observation is plainly the robot's camera image data
        observation = self.worldModule.envData['imageData']
        # display the observation camera image
        if self.visualOutput:
            imageData = observation
            self.cameraImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.cameraImage.setImage(imageData)
            imageScale = 1.0
            self.cameraImage.setRect(QRectF(0.0, 0.0, imageScale, imageData.shape[0]/imageData.shape[1]*imageScale))
        # scale the one-line image to further reduce computational demands
        observation = cv2.resize(observation, dsize=self.imageDims)
        observation.astype('float32')
        observation = observation/255.0
        # display the observation camera image reduced to one line
        if self.visualOutput:
            imageData = observation
            self.observationImage.setOpts(axisOrder='row-major')
            imageData = imageData[:,:,::-1]
            self.observationImage.setImage(imageData)
            imageScale = 1.0
            self.observationImage.setRect(QRectF(0.0, -0.1, imageScale, imageData.shape[0]/imageData.shape[1]*imageScale))
        self.observation = observation
        
    def getObservationSpace(self):
        '''
        This function returns the observation space for the given observation class.
        '''
        # currently, use a one-line 'image' to save computational resources
        return gym.spaces.Box (low=0.0, high=1.0, shape=(self.imageDims[1], self.imageDims[0], 3))
    
    
class ImageObservationUnity():
    '''
    This module computes an observation based on the current camera image acquired by the robot/agent.
    Raw image data is processed to line image observations.
    
    | **Args**
    | world:                        The world module.
    | guiParent:                    The main window for visualization.
    | visualOutput:                 Does the module provide visual output?
    | use_gray_scale:               If true, RGB images will be converted into gray scale images.
    | imageDims:                    The dimensions the observation will be resized to.
    '''
    
    def __init__(self, world, graphicsWindow, visualOutput=True, use_gray_scale=False, imageDims=(84, 84, 3)):

        # store the world module reference
        self.worldModule = world
        self.graphicsWindow = graphicsWindow
        self.topologyModule = None
        self.visualOutput = visualOutput
        self.observation = None
        self.use_gray_scale = use_gray_scale
        # generate a visual display of the observation
        if self.visualOutput:
            self.layout = self.graphicsWindow.centralWidget
            self.observation_plot_viewbox = qg.ViewBox(parent=self.layout, enableMouse=False, enableMenu=False)
            self.cameraImage = qg.ImageItem()
            # the observation plots will be initialized on receiving the
            self.layout.addItem(self.observation_plot_viewbox,
                                colspan=2, rowspan=1, row=1, col=0)

            self.observation_plot_viewbox.setAspectLocked(lock=True)
            self.observation_plot_viewbox.addItem(self.cameraImage)

        self.imageDims = imageDims
        if self.use_gray_scale:
            self.imageDims = (self.imageDims[0], self.imageDims[1], 1)

    def update(self):
        '''
        This function processes the raw image data and updates the current observation.
        '''
        # the robot's/agent's pose has changed, get the new observation by evaluating
        # information from the world module:
        # the observation is plainly the robot's camera image data
        observation = self.worldModule.envData['imageData']
        if self.use_gray_scale:
            observation = self.to_gray_scale_image(observation)
        observation = cv2.resize(observation, dsize=self.imageDims[:2])
        # display the observation camera image
        if self.visualOutput:
            imageData = observation
            self.cameraImage.setOpts(axisOrder='row-major')
            # mirror the image
            self.cameraImage.setImage(imageData[::-1])
            self.cameraImage.setRect(QRectF(0.0, 0.0, self.imageDims[0], self.imageDims[1]))
            self.cameraImage.setLevels([0.0, 0.9])
        self.observation = observation
    
    def getObservationSpace(self):
        '''
        This function returns the observation space for the given observation class.
        '''
        return gym.spaces.Box(low=0.0, high=1.0,shape=(self.imageDims[1], self.imageDims[0], self.imageDims[2]))

    def to_gray_scale_image(self, image_array):
        '''
        This function converts a 3D image array to a 2D grayscale image array.
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