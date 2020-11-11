



import sys
import socket
import os
import subprocess
import numpy as np


from shapely.geometry import Polygon



### The Blender interface class. This class connects to the Blender environment and controls the flow of commands/data that goes to/comes from the Blender environment.
###
class FrontendBlenderInterface():

    ### Constructor
    ### scenarioName: the FULL path to the scenario that should be processed
    def __init__(self, scenarioName):


        path=os.environ['BLENDER_EXECUTABLE_PATH']

        self.BLENDER_EXECUTABLE=path+'blender'

        paths=os.environ['PYTHONPATH'].split(';')
        path=None
        for p in paths:
            if 'CoBeL-RL' in p:
                full_path=p
                base_folder=full_path.split(sep='CoBeL-RL')[0]
                path=base_folder+'CoBeL-RL'
                break

        scenarioName=path+'/environments/environments_blender/'+scenarioName
        print(path)
        subprocess.Popen([self.BLENDER_EXECUTABLE,scenarioName,'--window-border','--window-geometry','1320','480','600','600','--enable-autoexec'])
        self.controlSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.videoSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.dataSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)



        # wait for BlenderControl to start, this socket take care of command/data(limited amount) transmission
        BlenderControlAvailable=False
        while not BlenderControlAvailable:
            try:
                self.controlSocket.connect(('localhost',5000))
                self.controlSocket.setblocking(1)
                BlenderControlAvailable=True
            except:
                pass

        print ('Blender control connection has been initiated.')
        self.controlSocket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

        # wait for BlenderVideo to start, this socket transmits video data from Blender
        BlenderVideoAvailable=False
        while not BlenderVideoAvailable:
            try:
                self.videoSocket.connect(('localhost',5001))
                self.videoSocket.setblocking(1)
                BlenderVideoAvailable=True
            except:
                pass

        print ('Blender video connection has been initiated.')
        self.videoSocket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)


        # wait for BlenderData to start, this socket is currently legacy, can probably be removed in future system versions(?)
        BlenderDataAvailable=False
        while not BlenderDataAvailable:
            try:
                self.dataSocket.connect(('localhost',5002))
                self.dataSocket.setblocking(1)
                BlenderDataAvailable=True
            except:
                pass

        print ('Blender data connection has been initiated.')
        self.dataSocket.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)

        # sockets are now open!

        # get the maximum safe zone dimensions for the robot to move in
        self.controlSocket.send('getSafeZoneDimensions'.encode('utf-8'))
        value=self.controlSocket.recv(1000).decode('utf-8')
        [minXStr,minYStr,minZStr,maxXStr,maxYStr,maxZStr]=value.split(',')
        # store the environment limits
        self.minX=float(minXStr)
        self.minY=float(minYStr)
        self.minZ=float(minZStr)

        self.maxX=float(maxXStr)
        self.maxY=float(maxYStr)
        self.maxZ=float(maxZStr)


        # get the safe zone layout for the robot to move in
        self.controlSocket.send('getSafeZoneLayout'.encode('utf-8'))
        value=self.controlSocket.recv(1000).decode('utf-8')
        self.controlSocket.send('AKN'.encode('utf-8'))
        nrOfVertices=int(value)

        # temporary variables for extraction of the environment's perimeter
        vertexList=[]
        safeZoneSegmentList=[]

        for i in range(nrOfVertices):
            value=self.controlSocket.recv(1000).decode('utf-8')
            xStr,yStr=value.split(',')
            x=float(xStr)
            y=float(yStr)
            self.controlSocket.send('AKN'.encode('utf-8'))
            # update the vertex list
            vertexList=vertexList+[[x,y]]
            j=i+1
            if i==nrOfVertices-1:
                j=0
            # update the segment list
            safeZoneSegmentList+=[[i,j]]

        # convert the above lists to numpy arrays
        self.safeZoneVertices=np.array(vertexList)
        self.safeZoneSegments=np.array(safeZoneSegmentList)


        # construct the safe polygon from the above lists
        self.safeZonePolygon=Polygon(vertexList)
        self.controlSocket.recv(100).decode('utf-8')



        # initial robot pose
        self.robotPose=np.array([0.0,0.0,1.0,0.0])

        # stores the actual goal position (coordinates of the goal node)
        self.goalPosition=None

        # indicator that flags arrival of the agent/robot at the actual goal node (this need NOT be the global goal node!)
        self.goalReached=False

        # a dict that stores environmental information in each time step
        self.envData=dict()
        self.envData['timeData']=None
        self.envData['poseData']=None
        self.envData['sensorData']=None
        self.envData['imageData']=None

        # propel the simulation for 10 timesteps to finish initialization of the simulation framework
        for i in range(10):
            self.step_simulation_without_physics(0.0,0.0,0.0)


    ### This function reads all manually defined topology nodes from the environment (if such nodes are defined)
    def getManuallyDefinedTopologyNodes(self):
        # get the forbidden zones' layouts for the robot to avoid
        self.controlSocket.send('getManuallyDefinedTopologyNodes'.encode('utf-8'))
        nrOfNodesStr=self.controlSocket.recv(1000).decode('utf-8')

        nrOfNodes=int(nrOfNodesStr)

        print('Received %d manually defined topology nodes' % nrOfNodes)

        self.controlSocket.send('AKN'.encode('utf-8'))

        manuallyDefinedTopologyNodes=[]

        for n in range(nrOfNodes):

            nodeStr=self.controlSocket.recv(1000).decode('utf-8')
            print('reading manually defined node: %s' % nodeStr)

            nameStr,xStr,yStr,typeStr=nodeStr.split(',')
            nodeName=nameStr
            nodeX=float(xStr)
            nodeY=float(yStr)
            nodeType=typeStr
            self.controlSocket.send('AKN'.encode('utf-8'))

            manuallyDefinedTopologyNodes+=[[nodeName,nodeX,nodeY,nodeType]]


        # wait for AKN
        self.controlSocket.recv(1000).decode('utf-8')

        return manuallyDefinedTopologyNodes


    ### This function reads all manually defined topology edges from the environment (if such edges are defined)
    def getManuallyDefinedTopologyEdges(self):
        # get the forbidden zones' layouts for the robot to avoid
        self.controlSocket.send('getManuallyDefinedTopologyEdges'.encode('utf-8'))

        nrOfEdgesStr=self.controlSocket.recv(1000).decode('utf-8')
        nrOfEdges=int(nrOfEdgesStr)

        print('Received %d manually defined topology edges' % nrOfEdges)

        self.controlSocket.send('AKN'.encode('utf-8'))

        manuallyDefinedTopologyEdges=[]

        for n in range(nrOfEdges):

            edgeStr=self.controlSocket.recv(1000).decode('utf-8')
            print('reading manually defined edge: %s' % edgeStr)

            nameStr,firstStr,secondStr=edgeStr.split(',')
            edgeName=nameStr
            first=int(firstStr)
            second=int(secondStr)
            self.controlSocket.send('AKN'.encode('utf-8'))

            manuallyDefinedTopologyEdges+=[[edgeName,first,second]]


        # wait for AKN
        self.controlSocket.recv(1000).decode('utf-8')


        return manuallyDefinedTopologyEdges





    # This function reads an image chunk from a socket.
    #
    # s:        socket to read image from
    # imgSize:  the size of the image to read
    def recvImage(self,s,imgSize):

        # define buffer
        imgData=b''

        # define byte 'counter'
        receivedBytes=0

        # read the image in chunks
        while receivedBytes<imgSize:

            dataChunk=s.recv(imgSize-receivedBytes)
            if dataChunk=='':
                break
            receivedBytes+=len(dataChunk)
            imgData+=dataChunk

        # return image data
        return imgData


    # This function propels the simulation. It uses physics to guide the agent/robot with linear and angular velocities.
    #
    # velocityLinear:       the requested linear (translational) velocity of the agent/robot
    # omega:                the requested rotational (angular) velocity of the agent/robot
    #
    # Note: linear and angular velocities can be set simultaneously to force the robot to go in curved paths
    def stepSimulation(self,velocityLinear,omega):

        # the basic capture image size for the images taken by the robot's omnicam, the width of the omnicam image is actually 4*capAreaWidth
        capAreaWidth=64
        capAreaHeight=64

        # from the linear/angular velocities, compute the left and right wheel velocities
        [velocityLeft,velocityRight]=self.setRobotVelocities(velocityLinear,omega)

        # send the actuation command to the virtual robot/agent
        sendStr='stepSimulation,%f,%f' % (velocityLeft,velocityRight)
        self.controlSocket.send(sendStr.encode('utf-8'))


        # retrieve images from all cameras of the robot (front, left, right, back)
        imgFront=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPFront=np.fromstring(imgFront,dtype=np.uint8)
        imgNPFront=imgNPFront.reshape((capAreaHeight,capAreaWidth,4))

        imgLeft=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPLeft=np.fromstring(imgLeft,dtype=np.uint8)
        imgNPLeft=imgNPLeft.reshape((capAreaHeight,capAreaWidth,4))


        imgRight=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPRight=np.fromstring(imgRight,dtype=np.uint8)
        imgNPRight=imgNPRight.reshape((capAreaHeight,capAreaWidth,4))


        imgBack=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPBack=np.fromstring(imgBack,dtype=np.uint8)
        imgNPBack=imgNPBack.reshape((capAreaHeight,capAreaWidth,4))

        # and construct the omnicam image from the single images. Note: the images are just 'stitched' together, no distortion correction takes place (so far).
        imageData=np.hstack((imgNPFront,imgNPRight,imgNPBack,imgNPLeft))
        # center(?) the omnicam image
        imageData=np.roll(imageData,96,axis=1)
        # extract RGB information channels
        imageData=imageData[:,:,0:3]

        # retrieve telemetry data from the virtual agent/robot
        telemetryDataString=self.controlSocket.recv(2000).decode('utf-8')
        [timeDataStr,poseDataStr,sensorDataStr]=telemetryDataString.split(':')

        # extract time data from telemetry
        timeData=float(timeDataStr)

        # extract pose data from telemetry
        [x,y,nx,ny]=poseDataStr.split(',')
        poseData=[float(x),float(y),float(nx),float(ny)]

        # extract sensor data from telemetry
        [s0,s1,s2,s3,s4,s5,s6,s7]=sensorDataStr.split(',')
        sensorData=np.array([float(s0),float(s1),float(s2),float(s3),float(s4),float(s5),float(s6),float(s7)],dtype='float')

        # update robot's/agent's pose
        self.robotPose=np.array(poseData)

        # update environmental information
        self.envData['timeData']=timeData
        self.envData['poseData']=poseData
        self.envData['sensorData']=sensorData
        self.envData['imageData']=imageData


        # return acquired data
        return [timeData,poseData,sensorData,imageData]



    # This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x,y,yaw values.
    #
    # x:       the global x position to teleport to
    # y:       the global y position to teleport to
    # yaw:     the global yaw value to teleport to
    #
    def step_simulation_without_physics(self,newX,newY,newYaw):

        # the basic capture image size for the images taken by the robot's omnicam, the width of the omnicam image is actually 4*capAreaWidth
        capAreaWidth=64
        capAreaHeight=64

        # send the actuation command to the virtual robot/agent
        sendStr='stepSimNoPhysics,%f,%f,%f' % (newX,newY,newYaw)
        self.controlSocket.send(sendStr.encode('utf-8'))


        # retrieve images from all cameras of the robot (front, left, right, back)
        imgFront=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPFront=np.fromstring(imgFront,dtype=np.uint8)
        imgNPFront=imgNPFront.reshape((capAreaHeight,capAreaWidth,4))

        imgLeft=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPLeft=np.fromstring(imgLeft,dtype=np.uint8)
        imgNPLeft=imgNPLeft.reshape((capAreaHeight,capAreaWidth,4))


        imgRight=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPRight=np.fromstring(imgRight,dtype=np.uint8)
        imgNPRight=imgNPRight.reshape((capAreaHeight,capAreaWidth,4))


        imgBack=self.recvImage(self.videoSocket,capAreaWidth*capAreaHeight*4)
        imgNPBack=np.fromstring(imgBack,dtype=np.uint8)
        imgNPBack=imgNPBack.reshape((capAreaHeight,capAreaWidth,4))

        # and construct the omnicam image from the single images. Note: the images are just 'stitched' together, no distortion correction takes place (so far).
        imageData=np.hstack((imgNPFront,imgNPRight,imgNPBack,imgNPLeft))
        # center(?) the omnicam image
        imageData=np.roll(imageData,96,axis=1)
        # extract RGB information channels
        imageData=imageData[:,:,0:3]


        # retrieve telemetry data from the virtual agent/robot
        telemetryDataString=self.controlSocket.recv(2000).decode('utf-8')
        [timeDataStr,poseDataStr,sensorDataStr]=telemetryDataString.split(':')

        # extract time data from telemetry
        timeData=float(timeDataStr)

        # extract pose data from telemetry
        [x,y,nx,ny]=poseDataStr.split(',')
        poseData=[float(x),float(y),float(nx),float(ny)]

        # extract sensor data from telemetry
        [s0,s1,s2,s3,s4,s5,s6,s7]=sensorDataStr.split(',')
        sensorData=np.array([float(s0),float(s1),float(s2),float(s3),float(s4),float(s5),float(s6),float(s7)],dtype='float')

        # update robot's/agent's pose
        self.robotPose=np.array(poseData)


        # update environmental information
        self.envData['timeData']=timeData
        self.envData['poseData']=poseData
        self.envData['sensorData']=sensorData
        self.envData['imageData']=imageData


        # return acquired data
        return [timeData,poseData,sensorData,imageData]



    # This function actually actuates the agent/robot in the virtual environment.
    #
    # actuatorCommand:  the command that is used in the actuation process
    #
    def actuateRobot(self,actuatorCommand):

        # if the actuator command has more than 2 array entries, this is a teleport command, and will cause a teleport jump of the agent/robot (no physics support)
        if actuatorCommand.shape[0]>2:

            # call the teleportation routine
            [timeData,poseData,sensorData,imageData]=self.step_simulation_without_physics(actuatorCommand[0],actuatorCommand[1],actuatorCommand[2])
            
            # flag if the robot reached the goal (should always be the case after a teleportation)
            if self.goalPosition is not None:
                print(np.linalg.norm(poseData[0:2]-self.goalPosition))
                if np.linalg.norm(poseData[0:2]-self.goalPosition)<0.01:
                    self.goalReached=True

            # return the data acquired from the robot/agent/environment
            return timeData,poseData,sensorData,imageData

        else:
            # otherwise, this is a standard motion command with physics support (the agent/robot approaches the goal by actuating the robot's/agent's wheels)
            [timeData,poseData,sensorData,imageData]=self.stepSimulation(actuatorCommand[0],actuatorCommand[1])

            # flag if the robot/agent reached the goal already
            if self.goalPosition is not None:
                if np.linalg.norm(poseData[0:2]-self.goalPosition)<0.01:
                    self.goalReached=True

            # return the data acquired from the robot/agent/environment
            return timeData,poseData,sensorData,imageData




    # This function teleports the robot to a novel pose.
    #
    # pose:     the pose to teleport to
    def setXYYaw(self,objectName,pose):

        # send the request for teleportation to Blender
        poseStr='%f,%f,%f' % (pose[0],pose[1],pose[2])
        sendStr='setXYYaw,robotSupport,%s' % poseStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)


    # This function sets the color of a dedicated light source
    #
    # lightSource:     the name of the light source to change
    # color:           the color of the light source (as [red,green,blue])

    def setIllumination(self,lightSource,color):

        # send the request for illumination change to Blender
        illuminationStr='%s,%f,%f,%f' % (lightSource,color[0],color[1],color[2])
        sendStr='setIllumination,%s' % illuminationStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)


    # This function returns the limits of the environmental perimeter.
    def getLimits(self):
        return np.array( [  [self.minX,self.maxX],
                            [self.minY,self.maxY]])

    # This function returns the environmental perimeter by means of wall vertices/segments.
    def getWallGraph(self):
        return self.safeZoneVertices,self.safeZoneSegments

    # This function shuts down Blender.
    def stopBlender(self):
        try:
            self.controlSocket.send('stopSimulation'.encode('utf-8'))

        except:
            print(sys.exc_info()[1])





























'''
A very basic class for performing ABA renewal experiments from static image input. It is assumed that the agent sees in every node of a topology graph a static 360deg image of the environment.
Herein, rotation of the agent is not enabled. This setup accellerates standard ABA renewal experiments, since the Blender rendering 'overhead' is not reqired. It is necessary that Blender is run prior to
the application of this class, since the interface assumes a worldStruct data block in the 'world' directory of the experiment folder. This data is generated by a initial run of the getWorldInformation.py script in the experiment
folder.
'''
class WORLD_ABAFromImagesInterface(qt.QtCore.QObject):

    # send this signal when the robot pose changed
    sig_triggerVectorBasedPlanning=qt.QtCore.pyqtSignal(np.ndarray,np.ndarray)
    sig_robotPoseChanged = qt.QtCore.pyqtSignal( np.ndarray )
    
    
    ### Constructor
    ### 
    ### imageSetContextA: the image set used for context A
    ### imageSetContextB: the image set used for context B
    ### 
    def __init__(self,imageSetContextA='worldInfo/imagesContextA.npy',imageSetContextB='worldInfo/imagesContextB.npy'):
        
        
        qt.QtCore.QObject.__init__(self)
        
        # retrieve all image information from the 'worldInfo' directory
        self.imagesContextA=np.load(imageSetContextA)
        self.imagesContextB=np.load(imageSetContextB)
        
        # this is a ABA renewal specific variable that identifies the context for fetching the images from, will be altered when an appropriate 'setIllumination' command is issued.
        self.currentContext=None
        
        # variables and lists for the forbidden zones
        self.forbiddenZonesPolygons=[]
        self.activationOfForbiddenZones=[]
        
        safeZoneDimensions=np.load('worldInfo/safeZoneDimensions.npy')
        self.minX=safeZoneDimensions[0]
        self.minY=safeZoneDimensions[1]
        self.minZ=safeZoneDimensions[2]
        
        self.maxX=safeZoneDimensions[3]
        self.maxY=safeZoneDimensions[4]
        self.maxZ=safeZoneDimensions[5]
        
        self.safeZoneVertices=np.load('worldInfo/safeZoneVertices.npy')
        self.safeZoneSegments=np.load('worldInfo/safeZoneSegments.npy')
        
        # initial robot pose
        self.robotPose=np.array([0.0,0.0,1.0,0.0])
        
        # stores the actual goal position (coordinates of the goal node)
        self.goalPosition=None
        
        # indicator that flags arrival of the agent/robot at the actual goal node (this need NOT be the global goal node!)
        self.goalReached=False
        
        # a dict that stores environmental information in each time step
        self.envData=dict()
        self.envData['timeData']=0.0
        self.envData['poseData']=None
        self.envData['sensorData']=None
        self.envData['imageData']=None
        
        # this interface class requires a topologyModule
        self.topologyModule=None
    
    
    ### This function sets the activation state of the forbidden zones
    ### 
    ### indexOfZone:        the index of the zone to be changed w.r.t. its activation state
    ### activationState:    the new activation state for the zone
    ### 
    def setActivationOfForbiddenZones(self,indexOfZone,activationState):
        self.activationOfForbiddenZones[indexOfZone]=activationState
    
    ### This function reads all forbidden zones from the environment and transfers them to the world module
    def readForbiddenZones(self):
        self.forbiddenZonesPolygons=np.load('worldInfo/forbiddenZonesPolygons.npy')
        for i in range(self.forbiddenZonesPolygons.shape[0]):
            self.activationOfForbiddenZones+=[True]
    
    
    
    ### This function supplies the interface with a valid topology module
    ### 
    ### topologyModule: the topologyModule to be supplied
    def setTopology(self,topologyModule):
        self.topologyModule=topologyModule

    ### This function reads all manually defined topology nodes from the environment (if such nodes are defined)
    def getManuallyDefinedTopologyNodes(self):
        
        manuallyDefinedTopologyNodes=np.load('worldInfo/topologyNodes.npy')
        return manuallyDefinedTopologyNodes
        
    ### This function reads all manually defined topology nodes from the environment (if such nodes are defined)
    def getManuallyDefinedTopologyEdges(self):
        
        manuallyDefinedTopologyEdges=np.load('worldInfo/topologyEdges.npy')
        return manuallyDefinedTopologyEdges
        
        
    '''
    This function allows to control the context state in the simulation.
    context:    the context that should be employed
    '''
    def setContext(self,context):
        self.currentContext=context
        self.stepSimNoPhysics(0.0,0.0,0.0)
                
        
    # This function propels the simulation. It uses teleportation to guide the agent/robot directly by means of global x,y,yaw values.
    # 
    # x:       the global x position to teleport to
    # y:       the global y position to teleport to
    # yaw:     the global yaw value to teleport to
    # 
    def stepSimNoPhysics(self,newX,newY,newYaw):
        
        
            
        
        # update robot's/agent's pose
        self.robotPose=np.array([newX,newY,newYaw])
        
        # update environmental information
        # propel the simulation time by 1/100 of a second (standard time step)
        self.envData['timeData']+=0.01
        # the position can be updated instantaneously
        self.envData['poseData']=np.array([newX,newY,np.cos(newYaw/180.0*np.pi),np.sin(newYaw/180.0*np.pi)])
        # there will be no need for sensor data in this interface class
        self.envData['sensorData']=np.array([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        
        
        # get the images from the 'worldInfo' directory
        imageData=None  
        
        if self.currentContext is 'A':
            imageData=self.imagesContextA[self.topologyModule.nextNode]
        
        if self.currentContext is 'B':
            imageData=self.imagesContextB[self.topologyModule.nextNode]
        
        
        
        # the image data is read from the 'worldInfo' directory
        self.envData['imageData']=imageData
        
        # signal that the robot's/agent's pose has changed
        self.sig_robotPoseChanged.emit(self.envData['poseData'])
        
        
        # return acquired data
        return [self.envData['timeData'],self.envData['poseData'],self.envData['sensorData'],self.envData['imageData']]
    


    # This function actually actuates the agent/robot in the virtual environment.
    # 
    # actuatorCommand:  the command that is used in the actuation process
    # 
    def actuateRobot(self,actuatorCommand):
        
        # if the actuator command has more than 2 array entries, this is a teleport command, and will cause a teleport jump of the agent/robot (no physics support)
        if actuatorCommand.shape[0]>2:
            # call the teleportation routine
            [timeData,poseData,sensorData,imageData]=self.stepSimNoPhysics(actuatorCommand[0],actuatorCommand[1],90.0)
            # flag if the robot reached the goal (should always be the case after a teleportation)
            if self.goalPosition is not None:
                if np.linalg.norm(poseData[0:2]-self.goalPosition)<0.01:
                    self.goalReached=True
            
            
            # return the data acquired from the robot/agent/environment 
            return timeData,poseData,sensorData,imageData
        
        else:
            # otherwise, this is a standard motion command with physics support (the agent/robot approaches the goal by actuating the robot's/agent's wheels)
            [timeData,poseData,sensorData,imageData]=self.stepSimulation(actuatorCommand[0],actuatorCommand[1])
            
            # flag if the robot/agent reached the goal already
            if self.goalPosition is not None:
                if np.linalg.norm(poseData[0:2]-self.goalPosition)<0.01:
                    self.goalReached=True
            
            # return the data acquired from the robot/agent/environment
            return timeData,poseData,sensorData,imageData




    # This function teleports the robot to a novel pose.
    # 
    # pose:     the pose to teleport to
    def setXYYaw(self,objectName,pose):
        
        # send the request for teleportation to Blender
        poseStr='%f,%f,%f' % (pose[0],pose[1],pose[2])
        sendStr='setXYYaw,robotSupport,%s' % poseStr
        self.controlSocket.send(sendStr.encode('utf-8'))
        # wait for acknowledge from Blender
        self.controlSocket.recv(50)
        
    
    # This function sets the color of a dedicated light source
    # 
    # lightSource:     the name of the light source to change
    # color:           the color of the light source (as [red,green,blue])
    
    def setIllumination(self,lightSource,color):
        # note that in this interface class, it is expected that 2 contexts (A and B) are encountered, thus it is sufficient to toggle contexts whenever a new 'setIllumination' command is received, using a state
        # change method:
        if self.currentContext==None:
            # so far, there is no context, this means the simulation has just started, and context A can be set
            self.currentContext='A'
            
        else: 
            if self.currentContext is 'A':
                # the simulation encounterd context A, now toggle to context B
                self.currentContext='B'
            
            else:
                if self.currentContext is 'B':
                    # the simulation encountered context B, now toggle back to context A
                    self.currentContext='A'
            
                    # after the third 'setIllumination' command, there would be a toogling back to context A again, yet this is not to happen in the targeted ABA renewal paradigm
    
    
        
    
    # This function actually triggers the control path. The routine itself is triggered when the world is time-stepped.
    def triggerControlPath(self):
        self.sig_triggerVectorBasedPlanning.emit(self.robotPose,self.goalPosition)
    
    
    
    
    
    # This function returns the limits of the environmental perimeter.
    def getLimits(self):
        return np.array( [  [self.minX,self.maxX],
                            [self.minY,self.maxY]])
    
    # This function returns the environmental perimeter by means of wall vertices/segments.
    def getWallGraph(self):
        return self.safeZoneVertices,self.safeZoneSegments
    
   




