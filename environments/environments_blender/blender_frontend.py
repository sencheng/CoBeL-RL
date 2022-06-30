# basic imports
import math
import socket
import numpy as np
import time
from time import sleep
# blender imports
import bge
import bgl
import bpy
import PhysicsConstraints
import GameLogic as gl
import VideoTexture
import mathutils
from mathutils import Vector
from bge import texture


class BlenderFrontend():
    '''
    Basic frontend.
    
    | **Args**
    | control_buffer_size:          The buffer size of the control connection when receiving commands from the framework.
    '''
    
    def __init__(self, control_buffer_size=100):
        # initialize variables
        # port/address for controlling the simulation
        self.CONTROL_IP_ADDRESS = '127.0.0.1'
        self.CONTROL_PORT = 5000
        # port/address for the transfer of captured images
        self.VIDEO_IP_ADDRESS = '127.0.0.1'
        self.VIDEO_PORT = 5001
        # port/address for the transfer of sensor data, etc.
        self.DATA_IP_ADDRESS = '127.0.0.1'
        self.DATA_PORT = 5002
        # define camera resolution
        self.capAreaWidth = 64
        self.capAreaHeight = 64
        # buffer size of the control connection
        self.control_buffer_size = control_buffer_size
        # tell wether or not the network has to become active
        self.NETWORK_REQUIRED = True
        # the simulation time counter
        self.simulationTime = 0.0
        # the timestep used in the simulation
        self.dT = 0.01
        # set physics figures
        # linear velocity in m/s
        self.linearVelocity = 4.0/10.0
        # angulat velocity in deg/s
        self.angularVelocity = 90.0
        # do adaptation of velocities
        self.linVel = self.linearVelocity # a one-to-one relationship
        # the radius of the linear velocity injection points
        self.rO = 0.5
        # the radius of the virtual wheels
        self.rI = 0.05
        # compute necessary angular velocities for the robot
        self.angVel = self.angularVelocity / 360.0 * (2.0 * np.pi * self.rO)
        # get main scene
        self.scene = bge.logic.getCurrentScene()
        # instantiate access to the single components of the scene
        self.controller = self.scene.objects['simulationBaseline']
        self.robotSupport = self.scene.objects['robotSupport']
        self.leftWheel = self.scene.objects['leftWheel']
        self.rightWheel = self.scene.objects['rightWheel']
        # instantiate sensors
        self.sensorForward = self.robotSupport.sensors['sensorForward']
        self.sensorLeft = self.robotSupport.sensors['sensorLeft']
        self.sensorRight = self.robotSupport.sensors['sensorRight']
        self.sensorBackward = self.robotSupport.sensors['sensorBackward']
        self.sensorArray = np.zeros(8, dtype='float')
        # canvasses
        self.canvasFront = self.scene.objects['canvasFront']
        self.canvasLeft = self.scene.objects['canvasLeft']
        self.canvasRight = self.scene.objects['canvasRight']
        self.canvasBack = self.scene.objects['canvasBack']
        # cameras
        self.cameraFront = self.scene.objects['camRobotFront']
        self.cameraLeft = self.scene.objects['camRobotLeft']
        self.cameraRight = self.scene.objects['camRobotRight']
        self.cameraBack = self.scene.objects['camRobotBack']
        # prepare canvases for image transfer    
        self.IDFront = texture.materialID(self.canvasFront, 'MAscreenFront')
        self.IDLeft = texture.materialID(self.canvasLeft, 'MAscreenLeft')
        self.IDRight = texture.materialID(self.canvasRight, 'MAscreenRight')
        self.IDBack = texture.materialID(self.canvasBack, 'MAscreenBack')
        # front canvas
        self.canvasFront['canvasTextureFront'] = texture.Texture(self.canvasFront, self.IDFront)
        self.canvasFront['canvasTextureFront'].source = texture.ImageRender(self.scene, self.cameraFront)
        self.canvasFront['canvasTextureFront'].source.capsize = [self.capAreaWidth, self.capAreaHeight]
        # left canvas
        self.canvasLeft['canvasTextureLeft'] = texture.Texture(self.canvasLeft, self.IDLeft)
        self.canvasLeft['canvasTextureLeft'].source = texture.ImageRender(self.scene, self.cameraLeft)
        self.canvasLeft['canvasTextureLeft'].source.capsize = [self.capAreaWidth, self.capAreaHeight]
        # right canvas
        self.canvasRight['canvasTextureRight'] = texture.Texture(self.canvasRight, self.IDLeft)
        self.canvasRight['canvasTextureRight'].source = texture.ImageRender(self.scene, self.cameraRight)
        self.canvasRight['canvasTextureRight'].source.capsize = [self.capAreaWidth, self.capAreaHeight]
        # back canvas
        self.canvasBack['canvasTextureBack'] = texture.Texture(self.canvasBack, self.IDBack)
        self.canvasBack['canvasTextureBack'].source = texture.ImageRender(self.scene, self.cameraBack)
        self.canvasBack['canvasTextureBack'].source.capsize = [self.capAreaWidth, self.capAreaHeight]
        # prepare the buffers that store the single images
        self.bufFront = np.zeros(self.capAreaWidth * self.capAreaHeight * 4, dtype = 'uint8')
        self.bufLeft = np.zeros(self.capAreaWidth * self.capAreaHeight * 4, dtype = 'uint8')
        self.bufRight = np.zeros(self.capAreaWidth * self.capAreaHeight * 4, dtype = 'uint8')
        self.bufBack = np.zeros(self.capAreaWidth * self.capAreaHeight * 4, dtype = 'uint8')
        # engage control method
        self.controller['controlSocket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller['controlSocket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.controller['controlSocket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)    
        self.controller['controlSocket'].bind((self.CONTROL_IP_ADDRESS, self.CONTROL_PORT))
        self.controller['controlSocket'].listen(1)
        self.controller['controlConnection'], address = self.controller['controlSocket'].accept()
        print('Accepted control client from: ', address)
        # engage video transfer
        self.controller['videoSocket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller['videoSocket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.controller['videoSocket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.controller['videoSocket'].bind((self.VIDEO_IP_ADDRESS, self.VIDEO_PORT))
        self.controller['videoSocket'].listen(1)
        self.controller['videoConnection'], address = self.controller['videoSocket'].accept()
        print('Accepted video client from: ', address)
        # engage data transfer
        self.controller['dataSocket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.controller['dataSocket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.controller['dataSocket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.controller['dataSocket'].bind((self.DATA_IP_ADDRESS, self.DATA_PORT))
        self.controller['dataSocket'].listen(1)
        self.controller['dataConnection'], address = self.controller['dataSocket'].accept()
        print('Accepted data client from: ', address)
        # signal presence of a network connection
        self.controller['networkUp'] = True
        print('network is up')
        # we will control the simulation through an external clock!
        bge.logic.setUseExternalClock(True)
        # instantiate the time system
        bge.logic.setClockTime(self.simulationTime)
        # set time scale
        bge.logic.setTimeScale(1.0)
        # make buffer
        self.buffer = bgl.Buffer(bgl.GL_BYTE, [64 * 64 * 3])
        # image acquisition
        self.imageAcquisitionEnabled = False
        # define functions accessible during the main loop
        self.functions = {}
        self.functions['resetSimulation'] = self.resetSimulation
        self.functions['stepSimulation'] = self.stepSimulation
        self.functions['stepSimNoPhysics'] = self.stepSimNoPhysics
        self.functions['getGoalPosition'] = self.getGoalPosition
        self.functions['setVelocity'] = self.setVelocity
        self.functions['getObservation'] = self.getObservation
        self.functions['getSensorData'] = self.getSensorData
        self.functions['getSimulationTime'] = self.getSimulationTime
        self.functions['getSafeZoneDimensions'] = self.getSafeZoneDimensions
        self.functions['getSafeZoneLayout'] = self.getSafeZoneLayout
        self.functions['getForbiddenZonesLayouts'] = self.getForbiddenZonesLayouts
        self.functions['getVisibleObjects'] = self.getVisibleObjects
        self.functions['stopSimulation'] = self.stopSimulation
        self.functions['suspendDynamics'] = self.suspendDynamics
        self.functions['restoreDynamics'] = self.restoreDynamics
        self.functions['renderLine'] = self.renderLine
        self.functions['teleportObject'] = self.teleportObject
        self.functions['setXYYaw'] = self.setXYYaw
        self.functions['getManuallyDefinedTopologyNodes'] = self.getManuallyDefinedTopologyNodes
        self.functions['getManuallyDefinedTopologyEdges'] = self.getManuallyDefinedTopologyEdges
        self.functions['setIllumination'] = self.setIllumination

    def actuateRobot(self):
        '''
        This function actuates the robot within the scene.
        '''
        if self.sensorForward.positive:  
            self.leftWheel.setLinearVelocity([self.linVel, 0.0, 0.0], True)
            self.rightWheel.setLinearVelocity([self.linVel, 0.0, 0.0], True)
        
        if self.sensorBackward.positive:
            self.leftWheel.setLinearVelocity([-self.linVel, 0.0, 0.0], True)
            self.rightWheel.setLinearVelocity([-self.linVel, 0.0, 0.0], True)
                    
        if self.sensorLeft.positive:
            self.leftWheel.setLinearVelocity([-self.angVel, 0.0, 0.0], True)
            self.rightWheel.setLinearVelocity([self.angVel, 0.0, 0.0], True)
                
        if self.sensorRight.positive:
            self.leftWheel.setLinearVelocity([self.angVel, 0.0, 0.0], True)
            self.rightWheel.setLinearVelocity([-self.angVel, 0.0, 0.0], True)

    def querySensors(self):
        '''
        This function queries the sensors and sets the sensorArray.
        '''
        for i in range(8):
            # retrieve sensor from scene
            sensorString = 'sensorMount%.3d' % i
            obj = self.scene.objects[sensorString]
            # retrieve sensor's world orientation and position
            orientation = obj.worldOrientation
            R = np.array(orientation)
            zDir = [orientation[0][2], orientation[1][2], orientation[2][2]]
            posSensor = np.array([obj.worldPosition[0], obj.worldPosition[1], obj.worldPosition[2]])        
            # compute ray directions
            maxAngle = 45
            numRays = 1
            sensorRange = 0.3
            sensorRayDirections = np.linspace(-maxAngle/2.0, maxAngle/2.0, numRays) 
            sensorRayDirections = np.array([0.0], dtype='float')
            sumDistances = 0.0
            minDistance = 1000.0
            for phi in sensorRayDirections:
                direction = Vector([np.cos(phi/180.0 * np.pi), np.sin(phi/180.0 * np.pi), 0.0])
                R = orientation
                newDir = R * direction
                d = 0.0
                ret = obj.rayCast(posSensor + newDir, posSensor, sensorRange, '', 1, 0, 0)
                if ret[0] is not None:
                    posHit = np.array(ret[1])
                    d = np.linalg.norm(posHit - posSensor, 2)
                else:
                    d = sensorRange
                sumDistances += d
                if minDistance > d:
                    minDistance = d
                bge.render.drawLine(posSensor, posSensor + newDir * d, (0, 1, 0)) 
            sumDistances /= sensorRayDirections.shape[0]
            self.sensorArray[i]=sumDistances
            self.sensorArray[i]=minDistance

    def calculateWheelInjectedSpeeds(self, v, omega, b=1.0, r=0.02):
        '''
        This function calculates the injected linear left and right wheel speeds for
        the robot from the desired linear velocity (v) and the desired angular velocity (omega).
        Calculations partially learned from http://www.uta.edu/utari/acs/jmireles/Robotics/KinematicsMobileRobots.pdf .
        
        | **Args**
        | v:                            The desired linear velocity.
        | omega:                        The desired angular velocity.
        | b:                            The distance between the robot's wheels.
        | r:                            The radius of the robot's wheels.
        | learningRate:                 The learning rate with which experiences are updated.
        '''
        # set variables
        b, r = b, r
        # the desired wheel speeds
        vL = v - np.pi * b / 2.0 * omega / 180.0
        vR = v + np.pi * b / 2.0 * omega / 180.0
            
        if (omega > 0.0):
            vR = np.pi
        else:
            vL = 0.0
            vR = 0.0
                
        return (vL, vR)
    
    def refresh_canvases(self):
        '''
        This function refreshes the canvases (i.e. the current camera input).
        '''
        # refresh canvases
        self.canvasFront['canvasTextureFront'].source.refresh(self.bufFront, 'BGRA')
        self.canvasLeft['canvasTextureLeft'].source.refresh(self.bufLeft, 'BGRA')
        self.canvasRight['canvasTextureRight'].source.refresh(self.bufRight, 'BGRA')
        self.canvasBack['canvasTextureBack'].source.refresh(self.bufBack, 'BGRA')
    
    def main_loop(self):
        '''
        This is the blender frontend's main loop.
        '''
        # start blender frontend main loop
        while not bge.logic.NextFrame():
            # retrieve the robot's heading
            heading = [self.robotSupport.worldOrientation[0][0], self.robotSupport.worldOrientation[1][0]]
            # propel the robot
            self.actuateRobot()
            # do some time statistics
            realTime = bge.logic.getRealTime()
            clockTime = bge.logic.getClockTime()
            frameTime = bge.logic.getFrameTime()
            # refresh canvases
            self.refresh_canvases()
            
            if self.NETWORK_REQUIRED:
                # prepare command read    
                data = ''
                # if the network is up start listening
                if self.controller['networkUp'] == True:
                    # retrieve data string from port
                    data = self.controller['controlConnection'].recv(self.control_buffer_size).decode('utf-8').split(',')
                    # execute command if it exists
                    if data[0] in self.functions:
                        if len(data) > 1:
                            self.functions[data[0]](data[1:])
                        else:
                            self.functions[data[0]]()
                        
    def resetSimulation(self):
        '''
        This function resets the simulation.
        '''
        self.simulationTime = 0.0
        self.leftWheel.setLinearVelocity([0.0, 0.0, 0.0], True)
        self.rightWheel.setLinearVelocity([0.0, 0.0, 0.0], True)
        bge.logic.setClockTime(self.simulationTime)
        
    def stepSimulation(self, data):
        '''
        This function updates the robot's velocity and propels the simulation by one time step.
        
        | **Args**
        | velocityLeftStr:              The desired left wheel velocity.
        | velocityRightStr:             The desired right wheel velocity.
        '''
        # split data string
        velocityLeftStr, velocityRightStr = data
        # recover wheel velocities
        velocityLeft = float(velocityLeftStr)
        velocityRight = float(velocityRightStr)
        # update simulation time
        self.simulationTime += self.dT
        # set wheel velocities
        self.leftWheel.setLinearVelocity([velocityLeft, 0.0, 0.0], True)
        self.rightWheel.setLinearVelocity([velocityRight, 0.0, 0.0], True)
        # update BGE clock
        bge.logic.setClockTime(self.simulationTime)
        # retrieve headings
        headingX = self.robotSupport.worldOrientation[0][0]
        headingY = self.robotSupport.worldOrientation[1][0]
        # refresh canvases
        self.refresh_canvases()
        # send control data
        sendString = '%.5f:%.3f,%.3f,%.3f,%.3f:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (self.simulationTime, self.robotSupport.worldPosition[0], self.robotSupport.worldPosition[1], headingX, headingY, self.sensorArray[0], self.sensorArray[1], self.sensorArray[2], self.sensorArray[3], self.sensorArray[4], self.sensorArray[5], self.sensorArray[6], self.sensorArray[7])
        self.controller['controlConnection'].send(sendString.encode('utf-8'))
        # send video data
        self.controller['videoConnection'].send(self.bufFront)
        self.controller['videoConnection'].send(self.bufLeft)
        self.controller['videoConnection'].send(self.bufRight)
        self.controller['videoConnection'].send(self.bufBack)
        
    def stepSimNoPhysics(self, data):
        '''
        This function teleports the robot and propels the simulation by one time step.
        
        | **Args**
        | velocityLeftStr:              The desired left wheel velocity.
        | velocityRightStr:             The desired right wheel velocity.
        '''
        # split data string
        newXStr, newYStr, newYawStr = data
        # recover position and orientation
        newX, newY, newYaw = float(newXStr), float(newYStr), float(newYawStr)
        # update simulation time
        self.simulationTime += self.dT
        # retrieve the robot's current position
        currentX, currentY, currentZ = self.robotSupport.worldPosition[0], self.robotSupport.worldPosition[1], self.robotSupport.worldPosition[2]
        # switch off physics for object in teleport
        self.robotSupport.setLinearVelocity([0.0, 0.0, 0.0], False)
        self.robotSupport.setAngularVelocity([0.0, 0.0, 0.0], False)
        # tie wheels to the robot's support
        self.leftWheel.setParent(self.robotSupport)
        self.rightWheel.setParent(self.robotSupport)
        # update the robot's position
        self.robotSupport.worldPosition.x = newX
        self.robotSupport.worldPosition.y = newY
        # update the robot's orientation
        euler = mathutils.Euler((0.0, 0.0, newYaw / 180 * math.pi), 'XYZ')
        self.robotSupport.worldOrientation = euler.to_matrix()
        # untie the wheels from the robot's support
        self.leftWheel.removeParent()
        self.rightWheel.removeParent()
        # update BGE clock
        bge.logic.setClockTime(self.simulationTime)
        # retrieve headings
        headingX = self.robotSupport.worldOrientation[0][0]
        headingY = self.robotSupport.worldOrientation[1][0]
        # refresh canvases
        self.refresh_canvases()
        # send control data
        sendString = '%.5f:%.3f,%.3f,%.3f,%.3f:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (self.simulationTime, self.robotSupport.worldPosition[0], self.robotSupport.worldPosition[1], headingX, headingY, self.sensorArray[0], self.sensorArray[1], self.sensorArray[2], self.sensorArray[3], self.sensorArray[4], self.sensorArray[5], self.sensorArray[6], self.sensorArray[7])
        self.controller['controlConnection'].send(sendString.encode('utf-8'))
        # send video data
        self.controller['videoConnection'].send(self.bufFront)
        self.controller['videoConnection'].send(self.bufLeft)
        self.controller['videoConnection'].send(self.bufRight)
        self.controller['videoConnection'].send(self.bufBack)
        
    def getGoalPosition(self):
        '''
        This function sends back the goal's current position.
        '''
        # retrieve goal position
        goalProxy = self.scene.objects['goalProxy']
        goalPos = goalProxy.worldPosition
        sendString = '%.3f,%.3f' % (goalPos[0], goalPos[1])    
        print(sendString)
        # send control data
        self.controller['controlConnection'].send(sendString.encode('utf-8'))
        
    def setVelocity(self, data):
        '''
        This function sets the robot's linear and angular velocities.
        
        | **Args**
        | objectName:                   The object name (unused).
        | velLinStr:                    The desired robot's linear velocity.
        | velAngStr:                    The desired robot's angular velocity.
        '''
        # split data string
        objectName, velLinStr, velAngStr = data
        # recover linear and angular velocities
        velLin = float(velLinStr)
        velAng = float(velAngStr)
        # compute left and right wheel velocities
        vL = -velAng
        vR = velAng
        # set left and right wheel velocities
        self.leftWheel.setLinearVelocity([vL, 0.0, 0.0], True)
        self.rightWheel.setLinearVelocity([vR, 0.0, 0.0], True)
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def getObservation(self):
        '''
        This function sends back the current image observation.
        '''
        print('getting observation')
        # send video data
        self.controller['videoConnection'].send(self.bufFront)
        self.controller['videoConnection'].send(self.bufLeft)
        self.controller['videoConnection'].send(self.bufRight)
        self.controller['videoConnection'].send(self.bufBack)
        print('observation sent')
        
    def getSensorData(self):
        '''
        This function sends back sensor data.
        '''
        sensorString = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (self.sensorArray[0], self.sensorArray[1], self.sensorArray[2], self.sensorArray[3], self.sensorArray[4], self.sensorArray[5], self.sensorArray[6], self.sensorArray[7])
        # send control data
        self.controller['controlConnection'].send(sensorString.encode('utf-8'))
        
    def getSimulationTime(self):
        '''
        This function sends back the current simulation time.
        '''
        timeString = '%.5f' % self.simulationTime
        # send control data
        self.controller['controlConnection'].send(timeString.encode('utf-8'))
        
    def getSafeZoneDimensions(self):
        '''
        This function sends back the dimensions of the safe area.
        Note: currently, the safeZoneLayout MUST have a scale of [1., 1., 1.]!
        '''
        # retrieve layout from scene
        layout = self.scene.objects['safeZoneLayout']
        # define min and max values
        maxX, maxY, maxZ = -1000.0, -1000.0, -1000.0
        minX, minY, minZ = 1000.0, 1000.0, 1000.0
        # loop over all vertices
        for mesh in layout.meshes:
            for mi in range(len(mesh.materials)):
                for vi in range(mesh.getVertexArrayLength(mi)):
                    vertex = mesh.getVertex(mi, vi)
                    # find extremal values
                    minX = min(minX, vertex.x)            
                    minY = min(minY, vertex.y)
                    minZ = min(minZ, vertex.z)
                    maxX = max(maxX, vertex.x)            
                    maxY = max(maxY, vertex.y)
                    maxZ = max(maxZ, vertex.z)
        retStr = '%.5f,%.5f,%.5f,%.5f,%.5f,%.5f' % (minX, minY, minZ, maxX, maxY, maxZ)
        # send control data
        self.controller['controlConnection'].send(retStr.encode('utf-8'))
        
    def getSafeZoneLayout(self):
        '''
        This function sends back the safe area's layout.
        Note: currently, the safeZoneLayout MUST have a scale of [1., 1., 1.]!
        '''
        # retrieve layout from scene
        layout = self.scene.objects['safeZoneLayout']
        # retrieve vertices
        mesh = bpy.data.objects['safeZoneLayout'].data
        vertices = mesh.vertices
        retStr = '%.5d' % (len(vertices))
        # send control data
        self.controller['controlConnection'].send(retStr.encode('utf-8'))    
        # wait for the framework's answer
        self.controller['controlConnection'].recv(1000)
        indexList = []
        # loop over mesh
        for p in mesh.polygons:
            for li in range(p.loop_start, p.loop_start + p.loop_total):
                index = mesh.loops[li].vertex_index
                print(index)
                print(vertices)
                retStr = '%.5f,%.5f' % (vertices[index].co[0],vertices[index].co[1])
                # send control data
                self.controller['controlConnection'].send(retStr.encode('utf-8'))
                # wait for the framework's answer
                self.controller['controlConnection'].recv(1000)                                                        
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def getForbiddenZonesLayouts(self):
        '''
        This function sends back the forbidden zone's layout.
        '''
        # retrieve forbidden zones
        forbiddenZonesObjects = []
        for obj in self.scene.objects:
            if 'forbiddenZone' in obj.name:
                forbiddenZonesObjects = forbiddenZonesObjects + [obj]
        # send forbidden zones
        print('found %d forbidden zones' % len(forbiddenZonesObjects))
        retStr = '%.d' % len(forbiddenZonesObjects)
        # send control data
        self.controller['controlConnection'].send(retStr.encode('utf-8'))
        # wait for the framework's answer
        self.controller['controlConnection'].recv(1000)
        # loop over forbidden zones
        for obj in forbiddenZonesObjects:
            # send zone name
            name = obj.name
            retStr = '%s' % name
            # send control data
            self.controller['controlConnection'].send(retStr.encode('utf-8'))
            # wait for the framework's answer
            self.controller['controlConnection'].recv(1000)
            # retrieve zone's vertices
            mesh = bpy.data.objects[name].data
            vertices = mesh.vertices
            # send vertices
            retStr = '%.5d' % (len(vertices))
            # send control data
            self.controller['controlConnection'].send(retStr.encode('utf-8'))
            # wait for the framework's answer
            self.controller['controlConnection'].recv(1000)
            # loop over mesh
            for p in mesh.polygons:
                for li in range(p.loop_start, p.loop_start + p.loop_total):
                    # vertex world stuff happens
                    index = mesh.loops[li].vertex_index
                    vertex = vertices[index].co
                    bObj = bpy.data.objects[name]
                    vertexWorld = bObj.matrix_world * vertex
                    retStr = '%.5f,%.5f' % (vertexWorld[0], vertexWorld[1])
                    # send control data
                    self.controller['controlConnection'].send(retStr.encode('utf-8'))
                    # wait for the framework's answer
                    self.controller['controlConnection'].recv(1000)
                    
    def getVisibleObjects(self):
        '''
        This function sends back the names and positions of all navigation landmarks.
        '''
        # retrieve objects from scene
        objects = self.scene.objects
        objectsToSend = []
        # filter out non-visible objects
        for obj in objects:
            if 'visibleObject' in obj:
                objectsToSend.append(obj)
        # prepare control data string
        retStr = ''
        for i in range(len(objectsToSend) - 1):
            obj = objectsToSend[i]
            retStr += '%f,%f,%s;' % (obj.worldPosition[0], obj.worldPosition[1], obj.name)
        obj = objectsToSend[-1]
        retStr += '%f,%f,%s' % (obj.worldPosition[0], obj.worldPosition[1], obj.name)
        # send control data
        self.controller['controlConnection'].send(retStr.encode('utf-8'))
        
    def stopSimulation(self):
        '''
        This function ends the simulation.
        '''
        # end the simulation
        bge.logic.endGame() 
        # close connections
        self.controller['controlConnection'].close()
        self.controller['videoConnection'].close()
        self.controller['networkUp'] = False
        
    def suspendDynamics(self, data):
        '''
        This function suspends the dynamics for a given object.
        
        | **Args**
        | objectName:                   The object for which dynamic should be suspended.
        '''
        # split data string
        objectName = data
        # suspend the object's dynamics
        obj = self.scene.objects[objectName]
        obj.suspendDynamics()
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def restoreDynamics(self, data):
        '''
        This function restores the dynamics for a given object.
        '''
        # split data string
        objectName = data
        # restore the object's dynamics
        obj = self.scene.objects[objectName]
        obj.restoreDynamics()
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def renderLine(self, data):
        '''
        This function renders a line.
        
        | **Args**
        | fromXStr:                     The line's start's x coordinate.
        | fromYStr:                     The line's start's y coordinate.
        | fromZStr:                     The line's start's z coordinate.
        | toXStr:                       The line's end's x coordinate.
        | toYStr:                       The line's end's y coordinate.
        | toZStr:                       The line's end's z coordinate.
        '''
        # split data string
        fromXStr, fromYStr, toXStr, toYStr, fromZStr, toZStr = data
        # render line
        bge.render.drawLine(np.array([float(fromXStr), float(fromYStr), float(fromZStr)]), np.array([float(toXStr), float(toYStr), float(toZStr)]), (0, 1, 0)) 
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def teleportObject(self, data):
        '''
        This function teleports an object to a desired new pose.
        
        | **Args**
        | objectName:                   The object name.
        | xStr:                         The desired x position.
        | yStr:                         The desired y position.
        | zStr:                         The desired z position.
        | rollStr:                      The desired roll in degrees.
        | pitchStr:                     The desired pitch in degrees.
        | yawStr:                       The desired yaw in degrees.
        '''
        # split data string
        objectName, xStr, yStr, zStr, rollStr, pitchStr, yawStr = data
        # recover pose
        x, y, z = float(xStr), float(yStr), float(zStr)
        roll, pitch, yaw = float(rollStr), float(pitchStr), float(yawStr)
        # retrieve object from scene
        obj = self.scene.objects[objectName]
        # switch off physics for object in teleport
        obj.setLinearVelocity([0.0, 0.0, 0.0], False)
        obj.setAngularVelocity([0.0, 0.0, 0.0], False)
        # tie wheels to the object (?)
        self.leftWheel.setParent(obj)
        self.rightWheel.setParent(obj)
        # update the object's position
        obj.worldPosition.x = x
        obj.worldPosition.y = y
        obj.worldPosition.z = z
        # update the object's orientation
        euler = mathutils.Euler((roll / 180 * math.pi, pitch / 180 * math.pi, yaw / 180 * math.pi), 'XYZ')
        obj.worldOrientation = euler.to_matrix()
        # untie wheels from the object (?)
        self.leftWheel.removeParent()
        self.rightWheel.removeParent()
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def setXYYaw(self, data):
        '''
        This function teleports an object to a desired new position and yaw.
        
        | **Args**
        | objectName:                   The object name.
        | xStr:                         The desired x position.
        | yStr:                         The desired y position.
        | yawStr:                       The desired yaw in degrees.
        '''
        # split data string
        objectName, xStr, yStr, yawStr = data
        # recover pose
        x, y = float(xStr), float(yStr)
        roll, pitch, yaw = 0., 0., float(yawStr)
        # retrieve object from scene
        obj = self.scene.objects[objectName]
        # switch off physics for object in teleport
        obj.setLinearVelocity([0.0, 0.0, 0.0], False)
        obj.setAngularVelocity([0.0, 0.0, 0.0], False)
        # tie wheels to object (?)
        self.leftWheel.setParent(obj)
        self.rightWheel.setParent(obj)
        # update the object's position
        obj.worldPosition.x = x
        obj.worldPosition.y = y
        # update the object's orientation
        euler = mathutils.Euler((roll / 180 * math.pi, pitch / 180 * math.pi, yaw / 180 * math.pi), 'XYZ')
        obj.worldOrientation = euler.to_matrix()
        # untie wheels to object (?)
        self.leftWheel.removeParent()
        self.rightWheel.removeParent()
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
    def getManuallyDefinedTopologyNodes(self):
        '''
        This function sends back the manually defined topology nodes.
        '''
        # retrieve the manually defined topology nodes from scene
        manuallyDefinedTopologyNodes = []
        for obj in self.scene.objects:
            if 'graphNode' in obj.name:
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
        
    def setIllumination(self, data):
        '''
        This function sets desired RGB values for a given light source.
        
        | **Args**
        | lightSourceName:              The light source's name.
        | redStr:                       The desired red value.
        | greenStr:                     The desired green value.
        | blueStr:                      The desired blue value.
        '''
        # split data string
        lightSourceName, redStr, greenStr, blueStr = data
        # recover RGB values
        red, green, blue = float(redStr), float(greenStr), float(blueStr)
        # update the light source's RGB values
        obj = self.scene.objects[lightSourceName]
        obj.color = [red, green, blue]
        print('changed color of light: %s to [%f,%f,%f]' % (lightSourceName, red, green, blue))
        # send control data
        self.controller['controlConnection'].send('AKN.'.encode('utf-8'))