
# basic imports

import numpy as np
from copy import deepcopy
import shutil
import os
import time

import cv2

import matplotlib
# use a backend that does not require visual output
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

from shapely.geometry import Polygon,Point

# imports from Hippocampus system
from modules.base.WORLD_Modules import WORLD_BlenderInterface
from modules.base.TOP_Modules import TOP_ManualTopologyModule
from modules.base.OBS_Modules import OBS_ImageObservationModule

from aux.fileAccess import readExperimentalDesignFile



if __name__ == "__main__":
    
    # read the experimental design file
    experimentalDesign=readExperimentalDesignFile()
    
    # extract the environment name from the experiment design
    environmentName=experimentalDesign['environmentName']
    topologyType=experimentalDesign['topologyType']
    topologyCliqueSize=experimentalDesign['topologyCliqueSize']
    topologyStartNodes=experimentalDesign['topologyStartNodes']
    topologyGoalNodes=experimentalDesign['topologyGoalNodes']
    
    # set up the world module
    worldModule=WORLD_BlenderInterface(environmentName)
    
    # set up the observation module
    observationModule=OBS_ImageObservationModule(worldModule,None,False)
    
    # create the modules dict required for the construction of the topology module
    modules=dict()
    modules['worldModule']=worldModule
    modules['observationModule']=observationModule
    print(experimentalDesign)
    # set up the topology module
    if topologyType=='ManualTopology':
        topologyModule=TOP_ManualTopologyModule(modules,None,{'startNodes':topologyStartNodes,'goalNodes':topologyGoalNodes,'cliqueSize':topologyCliqueSize},False)
    else:
        print('severe error: wrong topology type chosen (must be \'ManualTopology\'), stopping...')
        exit()
    
    
    
    
    # delete previous directory contents
    os.system('rm -rf worldInfo')
    # re-create the worldInfo directory
    os.makedirs('worldInfo')
   
   
    # store referenceImages sampled images from contexts A and B
    
    # Note: context A is conditioning and ROF (both red light), context b is extinction (white light). The light colors can be changed on demand.

    # prepare context A
    # switch on white light
    worldModule.setIllumination('Spot',[1.0,0.0,0.0])
    worldModule.stepSimNoPhysics(0.0,0.0,0.0)
    observationModule.createReferenceImages(topologyModule)
    imagesContextA=np.array(observationModule.referenceImages)
    
    # clear referenceImages
    observationModule.referenceImages=[]
    np.save('worldInfo/imagesContextA.npy',imagesContextA)
    
    
    
    # prepare context B
    # switch on white light
    worldModule.setIllumination('Spot',[0.0,0.0,0.0])
    worldModule.stepSimNoPhysics(0.0,0.0,0.0)
    
    observationModule.createReferenceImages(topologyModule)
    imagesContextB=np.array(observationModule.referenceImages)
    
    # clear referenceImages
    observationModule.referenceImages=[]
    np.save('worldInfo/imagesContextB.npy',imagesContextB)
    
    # store environment information
    nodes=np.array(worldModule.getManuallyDefinedTopologyNodes())
    nodes=nodes[nodes[:,0].argsort()]
    edges=np.array(worldModule.getManuallyDefinedTopologyEdges())
    edges=edges[edges[:,0].argsort()]
    np.save('worldInfo/topologyNodes.npy',nodes)
    np.save('worldInfo/topologyEdges.npy',edges)
    
    # since this experiment has a shock zone, store the perimeters
    worldModule.readForbiddenZones()
    np.save('worldInfo/forbiddenZonesPolygons.npy',worldModule.forbiddenZonesPolygons)
    
    safeZoneDimensions=np.array([worldModule.minX,worldModule.minY,worldModule.minZ,worldModule.maxX,worldModule.maxY,worldModule.maxZ])
    np.save('worldInfo/safeZoneDimensions.npy',safeZoneDimensions)
    
    np.save('worldInfo/safeZonePolygon.npy',worldModule.safeZonePolygon)
    np.save('worldInfo/safeZoneVertices.npy',worldModule.safeZoneVertices)
    np.save('worldInfo/safeZoneSegments.npy',worldModule.safeZoneSegments)
    
    # with the environment information stored, create randomly sampled images for LDA analysis
    
    # the image set size
    setSize=500
    dimX=1
    dimY=30
    
    stdX=1.5
    stdY=1.5
    stdPhi=0.0
    
    
    
    
    # generate random images, this is done by uniform sampling over the safe zone
    imagesContextA=np.zeros((setSize,dimX,dimY,3),dtype=float)
    imagesContextB=np.zeros((setSize,dimX,dimY,3),dtype=float)

    # prepare context A
    # switch on red light
    worldModule.setIllumination('Spot',[1.0,0.0,0.0])
    worldModule.stepSimNoPhysics(0.0,0.0,0.0)
    minZoneX,minZoneY,maxZoneX,maxZoneY=worldModule.safeZonePolygon.bounds
    print(minZoneX,minZoneY,maxZoneX,maxZoneY)
    i=0
    allowedPointsA=[]
    while i<setSize:
        newX=np.random.uniform(minZoneX,maxZoneX)
        newY=np.random.uniform(minZoneY,maxZoneY)
        newPhi=90.0+(np.random.random()*2.0-1.0)*stdPhi
        
        if Point(newX,newY).within(worldModule.safeZonePolygon):
            allowedPointsA+=[[newX,newY]]
            worldModule.stepSimNoPhysics(newX,newY,newPhi)
            worldModule.stepSimNoPhysics(newX,newY,newPhi)
            observation=worldModule.envData['imageData']
            observation=observation.astype('float32')
            observation=observation/255.0
            observation=cv2.resize(observation,dsize=(dimY,dimX))
            imagesContextA[i]=observation
            i+=1
    allowedPointsA=np.array(allowedPointsA)
    plt.scatter(allowedPointsA[:,0],allowedPointsA[:,1],s=1,c='r')
    #plt.show()
    # prepare context B
    # switch on white light
    worldModule.setIllumination('Spot',[0.0,0.0,0.0])
    #worldModule.setObjectsVisibilities(['extraMazeCueCross','extraMazeCueCircle'],[False,True])
    #worldModule.setObjectsVisibilities(['floorZebraStripesCue','floorLinearStripesCue'],[False,True])
    worldModule.stepSimNoPhysics(0.0,0.0,0.0)
    i=0
    allowedPointsB=[]
    while i<setSize:
        
        newX=np.random.uniform(minZoneX,maxZoneX)
        newY=np.random.uniform(minZoneY,maxZoneY)
        newPhi=90.0+(np.random.random()*2.0-1.0)*stdPhi
        
        if Point(newX,newY).within(worldModule.safeZonePolygon):
            allowedPointsB+=[[newX,newY]]
            worldModule.stepSimNoPhysics(newX,newY,newPhi)
            worldModule.stepSimNoPhysics(newX,newY,newPhi)
            observation=worldModule.envData['imageData']
            observation=observation.astype('float32')
            observation=observation/255.0
            observation=cv2.resize(observation,dsize=(dimY,dimX))
            imagesContextB[i]=observation
            i+=1
    allowedPointsB=np.array(allowedPointsB)
    plt.scatter(allowedPointsB[:,0],allowedPointsB[:,1],s=1,c='g')
    #plt.show()
            
    # store the randomly sampled images
    np.save('worldInfo/randomImagesContextA.npy',imagesContextA)
    np.save('worldInfo/randomImagesContextB.npy',imagesContextB)
