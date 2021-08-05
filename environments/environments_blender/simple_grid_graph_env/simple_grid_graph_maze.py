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

# define basic constants
# port/address for controlling the simulation
CONTROL_IP_ADDRESS = '127.0.0.1'
CONTROL_PORT = 5000
# port/address for the transfer of captured images
VIDEO_IP_ADDRESS = '127.0.0.1'
VIDEO_PORT = 5001
# port/address for the transfer of sensor data, etc.
DATA_IP_ADDRESS = '127.0.0.1'
DATA_PORT = 5002

# define camera resolution
capAreaWidth = 64
capAreaHeight = 64

# tell wether or not the network has to become active
NETWORK_REQUIRED = True

# the simulation time counter
simulationTime = 0.0
# the timestep used in the simulation
dT = 0.01

# set physics figures
# linear velocity in m/s
linearVelocity = 4.0/10.0
# angulat velocity in deg/s
angularVelocity = 90.0
# do adaptation of velocities
linVel = linearVelocity # a one-to-one relationship
# the radius of the linear velocity injection points
rO = 0.5
# the radius of the virtual wheels
rI = 0.05
# compute necessary angular velocities for the robot
angVel = angularVelocity / 360.0 * (2.0 * np.pi * rO)

# get main scene
scene = bge.logic.getCurrentScene()

# instantiate access to the single components of the scene
controller = scene.objects['simulationBaseline']
robotSupport = scene.objects['robotSupport']
leftWheel = scene.objects['leftWheel']
rightWheel = scene.objects['rightWheel']

# instantiate sensors
sensorForward = robotSupport.sensors['sensorForward']
sensorLeft = robotSupport.sensors['sensorLeft']
sensorRight = robotSupport.sensors['sensorRight']
sensorBackward = robotSupport.sensors['sensorBackward']
sensorArray = np.zeros(8,dtype='float')

# canvasses
canvasFront = scene.objects['canvasFront']
canvasLeft = scene.objects['canvasLeft']
canvasRight = scene.objects['canvasRight']
canvasBack = scene.objects['canvasBack']

# cameras
cameraFront = scene.objects['camRobotFront']
cameraLeft = scene.objects['camRobotLeft']
cameraRight = scene.objects['camRobotRight']
cameraBack = scene.objects['camRobotBack']


def actuateRobot():
    '''
    This function actuates the robot within the scene.
    '''
    if sensorForward.positive:  
        leftWheel.setLinearVelocity([linVel, 0.0, 0.0], True)
        rightWheel.setLinearVelocity([linVel, 0.0, 0.0], True)

    if sensorBackward.positive:
        leftWheel.setLinearVelocity([-linVel, 0.0, 0.0], True)
        rightWheel.setLinearVelocity([-linVel, 0.0, 0.0], True)
            
    if sensorLeft.positive:
        leftWheel.setLinearVelocity([-angVel, 0.0, 0.0], True)
        rightWheel.setLinearVelocity([angVel, 0.0, 0.0], True)
        
    if sensorRight.positive:
        leftWheel.setLinearVelocity([angVel, 0.0, 0.0], True)
        rightWheel.setLinearVelocity([-angVel, 0.0, 0.0], True)

def querySensors():
    '''
    This function queries the sensors and sets the sensorArray.
    '''
    for i in range(8):
        # retrieve sensor from scene
        sensorString = 'sensorMount%.3d' % i
        obj = scene.objects[sensorString]
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
        sensorArray[i]=sumDistances
        sensorArray[i]=minDistance

def calculateWheelInjectedSpeeds(v, omega, b=1.0, r=0.02):
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
    b = b
    r = r
    # the desired wheel speeds
    vL = v - np.pi * b / 2.0 * omega / 180.0
    vR = v + np.pi * b / 2.0 * omega / 180.0
    
    if (omega > 0.0):
        vR = np.pi
    else:
        vL = 0.0
        vR = 0.0
        
    return (vL, vR)


# prepare canvases for image transfer    
IDFront = texture.materialID(canvasFront, 'MAscreenFront')
IDLeft = texture.materialID(canvasLeft, 'MAscreenLeft')
IDRight = texture.materialID(canvasRight, 'MAscreenRight')
IDBack = texture.materialID(canvasBack, 'MAscreenBack')
# front canvas
canvasFront['canvasTextureFront'] = texture.Texture(canvasFront, IDFront)
canvasFront['canvasTextureFront'].source = texture.ImageRender(scene, cameraFront)
canvasFront['canvasTextureFront'].source.capsize = [capAreaWidth, capAreaHeight]
# left canvas
canvasLeft['canvasTextureLeft'] = texture.Texture(canvasLeft, IDLeft)
canvasLeft['canvasTextureLeft'].source = texture.ImageRender(scene, cameraLeft)
canvasLeft['canvasTextureLeft'].source.capsize = [capAreaWidth, capAreaHeight]
# right canvas
canvasRight['canvasTextureRight'] = texture.Texture(canvasRight, IDLeft)
canvasRight['canvasTextureRight'].source = texture.ImageRender(scene, cameraRight)
canvasRight['canvasTextureRight'].source.capsize = [capAreaWidth, capAreaHeight]
# back canvas
canvasBack['canvasTextureBack'] = texture.Texture(canvasBack, IDBack)
canvasBack['canvasTextureBack'].source = texture.ImageRender(scene, cameraBack)
canvasBack['canvasTextureBack'].source.capsize = [capAreaWidth, capAreaHeight]
# prepare the buffers that store the single images
bufFront = np.zeros((capAreaWidth * capAreaHeight * 4), dtype = 'uint8')
bufLeft = np.zeros(capAreaWidth * capAreaHeight * 4, dtype = 'uint8')
bufRight = np.zeros(capAreaWidth * capAreaHeight * 4, dtype = 'uint8')
bufBack = np.zeros(capAreaWidth * capAreaHeight * 4, dtype = 'uint8')

# the following is only done if we have a network connection!
if NETWORK_REQUIRED:
    # engage control method
    controller['controlSocket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    controller['controlSocket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    controller['controlSocket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)    
    controller['controlSocket'].bind((CONTROL_IP_ADDRESS, CONTROL_PORT))
    controller['controlSocket'].listen(1)
    controller['controlConnection'], address = controller['controlSocket'].accept()
    print('Accepted control client from: ', address)
    # engage video transfer
    controller['videoSocket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    controller['videoSocket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    controller['videoSocket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    controller['videoSocket'].bind((VIDEO_IP_ADDRESS, VIDEO_PORT))
    controller['videoSocket'].listen(1)
    controller['videoConnection'], address = controller['videoSocket'].accept()
    print('Accepted video client from: ', address)
    # engage data transfer
    controller['dataSocket'] = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    controller['dataSocket'].setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    controller['dataSocket'].setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    controller['dataSocket'].bind((DATA_IP_ADDRESS, DATA_PORT))
    controller['dataSocket'].listen(1)
    controller['dataConnection'], address = controller['dataSocket'].accept()
    print('Accepted data client from: ', address)
    # signal presence of a network connection
    controller['networkUp'] = True
    print('network is up')

# we will control the simulation through an external clock!
bge.logic.setUseExternalClock(True)
# instantiate the time system
bge.logic.setClockTime(simulationTime)
# set time scale
bge.logic.setTimeScale(1.0)
# make buffer
buffer = bgl.Buffer(bgl.GL_BYTE, [64 * 64 * 3])
# image acquisition
imageAcquisitionEnabled = False

# start blender frontend main loop
while not bge.logic.NextFrame():
    # retrieve the robot's heading
    heading = [robotSupport.worldOrientation[0][0], robotSupport.worldOrientation[1][0]]
    # propel the robot
    actuateRobot()
    # do some time statistics
    realTime = bge.logic.getRealTime()
    clockTime = bge.logic.getClockTime()
    frameTime = bge.logic.getFrameTime()
    # refresh canvases
    canvasFront['canvasTextureFront'].source.refresh(bufFront, 'BGRA')
    canvasLeft['canvasTextureLeft'].source.refresh(bufLeft, 'BGRA')
    canvasRight['canvasTextureRight'].source.refresh(bufRight, 'BGRA')
    canvasBack['canvasTextureBack'].source.refresh(bufBack, 'BGRA')
    
    if NETWORK_REQUIRED:
        # prepare command read    
        data = ''
        # if the network is up start listening
        if controller['networkUp'] == True:
            # retrieve data string from port
            data = controller['controlConnection'].recv(100).decode('utf-8')
            
            # functions
            
            if 'resetSimulation' in data:
                '''
                This function resets the simulation.
                '''
                simulationTime = 0.0
                leftWheel.setLinearVelocity([0.0, 0.0, 0.0], True)
                rightWheel.setLinearVelocity([0.0, 0.0, 0.0], True)
                bge.logic.setClockTime(simulationTime)
                
            if 'stepSimulation' in data:
                '''
                This function updates the robot's velocity and propels the simulation by one time step.
                
                | **Args**
                | velocityLeftStr:              The desired left wheel velocity.
                | velocityRightStr:             The desired right wheel velocity.
                '''
                # split data string
                [command, velocityLeftStr, velocityRightStr] = data.split(',')
                # recover wheel velocities
                velocityLeft = float(velocityLeftStr)
                velocityRight = float(velocityRightStr)
                # update simulation time
                simulationTime += dT
                # set wheel velocities
                leftWheel.setLinearVelocity([velocityLeft, 0.0, 0.0], True)
                rightWheel.setLinearVelocity([velocityRight, 0.0, 0.0], True)
                # update BGE clock
                bge.logic.setClockTime(simulationTime)
                # retrieve headings
                headingX = robotSupport.worldOrientation[0][0]
                headingY = robotSupport.worldOrientation[1][0]
                # send control data
                sendString = '%.5f:%.3f,%.3f,%.3f,%.3f:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (simulationTime, robotSupport.worldPosition[0], robotSupport.worldPosition[1], headingX, headingY, sensorArray[0], sensorArray[1], sensorArray[2], sensorArray[3], sensorArray[4], sensorArray[5], sensorArray[6], sensorArray[7])
                controller['controlConnection'].send(sendString.encode('utf-8'))
                # send video data
                controller['videoConnection'].send(bufFront)
                controller['videoConnection'].send(bufLeft)
                controller['videoConnection'].send(bufRight)
                controller['videoConnection'].send(bufBack)
            
            if 'stepSimNoPhysics' in data:
                '''
                This function teleports the robot and propels the simulation by one time step.
                
                | **Args**
                | velocityLeftStr:              The desired left wheel velocity.
                | velocityRightStr:             The desired right wheel velocity.
                '''
                # split data string
                [command, newXStr, newYStr, newYawStr] = data.split(',')
                # recover position and orientation
                newX = float(newXStr)
                newY = float(newYStr)
                newYaw = float(newYawStr)
                # update simulation time
                simulationTime += dT
                # retrieve the robot's current position
                currentX = robotSupport.worldPosition[0]
                currentY = robotSupport.worldPosition[1]
                currentZ = robotSupport.worldPosition[2]
                # switch off physics for object in teleport
                robotSupport.setLinearVelocity([0.0, 0.0, 0.0], False)
                robotSupport.setAngularVelocity([0.0, 0.0, 0.0], False)
                # tie wheels to the robot's support
                leftWheel.setParent(robotSupport)
                rightWheel.setParent(robotSupport)
                # update the robot's position
                robotSupport.worldPosition.x = newX
                robotSupport.worldPosition.y = newY
                # update the robot's orientation
                euler = mathutils.Euler((0.0, 0.0, newYaw / 180 * math.pi), 'XYZ')
                robotSupport.worldOrientation = euler.to_matrix()
                # untie the wheels from the robot's support
                leftWheel.removeParent()
                rightWheel.removeParent()
                # update BGE clock
                bge.logic.setClockTime(simulationTime)
                # retrieve headings
                headingX = robotSupport.worldOrientation[0][0]
                headingY = robotSupport.worldOrientation[1][0]
                # send control data
                sendString = '%.5f:%.3f,%.3f,%.3f,%.3f:%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (simulationTime, robotSupport.worldPosition[0], robotSupport.worldPosition[1], headingX, headingY, sensorArray[0], sensorArray[1], sensorArray[2], sensorArray[3], sensorArray[4], sensorArray[5], sensorArray[6], sensorArray[7])
                controller['controlConnection'].send(sendString.encode('utf-8'))
                # send video data
                controller['videoConnection'].send(bufFront)
                controller['videoConnection'].send(bufLeft)
                controller['videoConnection'].send(bufRight)
                controller['videoConnection'].send(bufBack)
                
            if 'getGoalPosition' in data:
                '''
                This function sends back the goal's current position.
                '''
                # retrieve goal position
                goalProxy = scene.objects['goalProxy']
                goalPos = goalProxy.worldPosition
                sendString = '%.3f,%.3f' % (goalPos[0], goalPos[1])    
                print(sendString)
                # send control data
                controller['controlConnection'].send(sendString.encode('utf-8'))
                
            if 'setVelocity' in data:
                '''
                This function sets the robot's linear and angular velocities.
                
                | **Args**
                | objectName:                   The object name (unused).
                | velLinStr:                    The desired robot's linear velocity.
                | velAngStr:                    The desired robot's angular velocity.
                '''
                # split data string
                [command, objectName, velLinStr, velAngStr] = data.split(',')
                # recover linear and angular velocities
                velLin = float(velLinStr)
                velAng = float(velAngStr)
                # compute left and right wheel velocities
                vL = -velAng
                vR = velAng
                # set left and right wheel velocities
                leftWheel.setLinearVelocity([vL, 0.0, 0.0], True)
                rightWheel.setLinearVelocity([vR, 0.0, 0.0], True)
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
            
            if 'getObservation' in data:
                '''
                This function sends back the current image observation.
                '''
                print('getting observation')
                # send video data
                controller['videoConnection'].send(bufFront)
                controller['videoConnection'].send(bufLeft)
                controller['videoConnection'].send(bufRight)
                controller['videoConnection'].send(bufBack)
                print('observation sent')
            
            if 'getSensorData' in data:
                '''
                This function sends back sensor data.
                '''
                sensorString = '%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f' % (sensorArray[0], sensorArray[1], sensorArray[2], sensorArray[3], sensorArray[4], sensorArray[5], sensorArray[6], sensorArray[7])
                # send control data
                controller['controlConnection'].send(sensorString.encode('utf-8'))
                
            if 'getSimulationTime' in data:
                '''
                This function sends back the current simulation time.
                '''
                timeString = '%.5f' % simulationTime
                # send control data
                controller['controlConnection'].send(timeString.encode('utf-8'))
            
            if 'getSafeZoneDimensions' in data:
                '''
                This function sends back the dimensions of the safe area.
                Note: currently, the safeZoneLayout MUST have a scale of [1., 1., 1.]!
                '''
                # retrieve layout from scene
                layout = scene.objects['safeZoneLayout']
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
                controller['controlConnection'].send(retStr.encode('utf-8'))    
                
            if 'getSafeZoneLayout' in data:
                '''
                This function sends back the safe area's layout.
                Note: currently, the safeZoneLayout MUST have a scale of [1., 1., 1.]!
                '''
                # retrieve layout from scene
                layout = scene.objects['safeZoneLayout']
                # retrieve vertices
                mesh = bpy.data.objects['safeZoneLayout'].data
                vertices = mesh.vertices
                retStr = '%.5d' % (len(vertices))
                # send control data
                controller['controlConnection'].send(retStr.encode('utf-8'))    
                # wait for the framework's answer
                controller['controlConnection'].recv(1000)
                indexList = []
                # loop over mesh
                for p in mesh.polygons:
                    for li in range(p.loop_start, p.loop_start + p.loop_total):
                        index = mesh.loops[li].vertex_index
                        print(index)
                        print(vertices)
                        retStr = '%.5f,%.5f' % (vertices[index].co[0],vertices[index].co[1])
                        # send control data
                        controller['controlConnection'].send(retStr.encode('utf-8'))
                        # wait for the framework's answer
                        controller['controlConnection'].recv(1000)                                                        
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
                      
            if 'getForbiddenZonesLayouts' in data:
                '''
                This function sends back the forbidden zone's layout.
                '''
                # retrieve forbidden zones
                forbiddenZonesObjects = []
                for obj in scene.objects:
                    if 'forbiddenZone' in obj.name:
                        forbiddenZonesObjects = forbiddenZonesObjects + [obj]
                # send forbidden zones
                print('found %d forbidden zones' % len(forbiddenZonesObjects))
                retStr = '%.d' % len(forbiddenZonesObjects)
                # send control data
                controller['controlConnection'].send(retStr.encode('utf-8'))
                # wait for the framework's answer
                controller['controlConnection'].recv(1000)
                # loop over forbidden zones
                for obj in forbiddenZonesObjects:
                    # send zone name
                    name = obj.name
                    retStr = '%s' % name
                    # send control data
                    controller['controlConnection'].send(retStr.encode('utf-8'))
                    # wait for the framework's answer
                    controller['controlConnection'].recv(1000)
                    # retrieve zone's vertices
                    mesh = bpy.data.objects[name].data
                    vertices = mesh.vertices
                    # send vertices
                    retStr = '%.5d' % (len(vertices))
                    # send control data
                    controller['controlConnection'].send(retStr.encode('utf-8'))
                    # wait for the framework's answer
                    controller['controlConnection'].recv(1000)
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
                            controller['controlConnection'].send(retStr.encode('utf-8'))
                            # wait for the framework's answer
                            controller['controlConnection'].recv(1000)       
            
            if 'getVisibleObjects' in data:
                '''
                This function sends back the names and positions of all navigation landmarks.
                '''
                # retrieve objects from scene
                objects = scene.objects
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
                controller['controlConnection'].send(retStr.encode('utf-8'))    
                    
            if 'stopSimulation' in data:
                '''
                This function ends the simulation.
                '''
                # end the simulation
                bge.logic.endGame() 
                # close connections
                controller['controlConnection'].close()
                controller['videoConnection'].close()
                controller['networkUp'] = False
               
            if 'suspendDynamics' in data:
                '''
                This function suspends the dynamics for a given object.
                
                | **Args**
                | objectName:                   The object for which dynamic should be suspended.
                '''
                # split data string
                [command, objectName] = data.split(',')
                # suspend the object's dynamics
                obj = scene.objects[objectName]
                obj.suspendDynamics()
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
            
            if 'restoreDynamics' in data:
                '''
                This function restores the dynamics for a given object.
                '''
                # split data string
                [command, objectName] = data.split(',')
                # restore the object's dynamics
                obj = scene.objects[objectName]
                obj.restoreDynamics()
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
            
            if 'renderLine' in data:
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
                [command, fromXStr, fromYStr, toXStr, toYStr, fromZStr, toZStr] = data.split(',')
                # render line
                bge.render.drawLine(np.array([float(fromXStr), float(fromYStr), float(fromZStr)]), np.array([float(toXStr), float(toYStr), float(toZStr)]), (0, 1, 0)) 
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
                
            if 'teleportObject' in data:
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
                [command, objectName, xStr, yStr, zStr, rollStr, pitchStr, yawStr] = data.split(',')
                # recover pose
                x = float(xStr)
                y = float(yStr)
                z = float(zStr)
                roll = float(rollStr)
                pitch = float(pitchStr)
                yaw = float(yawStr)
                # retrieve object from scene
                obj = scene.objects[objectName]
                # switch off physics for object in teleport
                obj.setLinearVelocity([0.0, 0.0, 0.0], False)
                obj.setAngularVelocity([0.0, 0.0, 0.0], False)
                # tie wheels to the object (?)
                leftWheel.setParent(obj)
                rightWheel.setParent(obj)
                # update the object's position
                obj.worldPosition.x = x
                obj.worldPosition.y = y
                obj.worldPosition.z = z
                # update the object's orientation
                euler = mathutils.Euler((roll / 180 * math.pi, pitch / 180 * math.pi, yaw / 180 * math.pi), 'XYZ')
                obj.worldOrientation = euler.to_matrix()
                # untie wheels from the object (?)
                leftWheel.removeParent()
                rightWheel.removeParent()
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))

            if 'setXYYaw' in data:
                '''
                This function teleports an object to a desired new position and yaw.
                
                | **Args**
                | objectName:                   The object name.
                | xStr:                         The desired x position.
                | yStr:                         The desired y position.
                | yawStr:                       The desired yaw in degrees.
                '''
                # split data string
                [command, objectName, xStr, yStr, yawStr] = data.split(',')
                # recover pose
                x = float(xStr)
                y = float(yStr)
                roll = 0.0
                pitch = 0.0
                yaw = float(yawStr)
                # retrieve object from scene
                obj = scene.objects[objectName]
                # switch off physics for object in teleport
                obj.setLinearVelocity([0.0, 0.0, 0.0], False)
                obj.setAngularVelocity([0.0, 0.0, 0.0], False)
                # tie wheels to object (?)
                leftWheel.setParent(obj)
                rightWheel.setParent(obj)
                # update the object's position
                obj.worldPosition.x = x
                obj.worldPosition.y = y
                # update the object's orientation
                euler = mathutils.Euler((roll / 180 * math.pi, pitch / 180 * math.pi, yaw / 180 * math.pi), 'XYZ')
                obj.worldOrientation = euler.to_matrix()
                # untie wheels to object (?)
                leftWheel.removeParent()
                rightWheel.removeParent()
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
                 
            if 'getManuallyDefinedTopologyNodes' in data:
                '''
                This function sends back the manually defined topology nodes.
                '''
                # retrieve the manually defined topology nodes from scene
                manuallyDefinedTopologyNodes = []
                for obj in scene.objects:
                    if 'graphNode' in obj.name:
                        manuallyDefinedTopologyNodes += [obj]
                # send the number of nodes
                print('found %d manually defined topology nodes' % len(manuallyDefinedTopologyNodes))
                retStr = '%.d' % len(manuallyDefinedTopologyNodes)
                # send control data
                controller['controlConnection'].send(retStr.encode('utf-8'))
                # wait for the framework's answer
                controller['controlConnection'].recv(1000)
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
                    controller['controlConnection'].send(retStr.encode('utf-8'))
                    # wait for the framework's answer
                    controller['controlConnection'].recv(1000)        
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
        
            if 'getManuallyDefinedTopologyEdges' in data:
                '''
                This function sends back the manually defined topology edges.
                '''
                # retrieve the manually defined topology edges from scene
                manuallyDefinedTopologyEdges = []
                for obj in scene.objects:
                    if 'graphEdge' in obj.name:
                        manuallyDefinedTopologyEdges += [obj]
                # send the number of edges
                print('found %d manually defined topology edges' % len(manuallyDefinedTopologyEdges))
                retStr = '%.d' % len(manuallyDefinedTopologyEdges)
                # send control data
                controller['controlConnection'].send(retStr.encode('utf-8'))
                # wait for the framework's answer
                controller['controlConnection'].recv(1000)
                # loop over edges
                for obj in manuallyDefinedTopologyEdges:
                    # retrieve edge information
                    name = obj.name
                    first = obj['first']
                    second = obj['second']
                    # send edge information
                    retStr = '%s,%d,%d' % (name, first, second)
                    # send control data
                    controller['controlConnection'].send(retStr.encode('utf-8'))
                    # wait for the framework's answer
                    controller['controlConnection'].recv(1000)
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))
                 
            if 'setIllumination' in data:
                '''
                This function sets desired RGB values for a given light source.
                
                | **Args**
                | lightSourceName:              The light source's name.
                | redStr:                       The desired red value.
                | greenStr:                     The desired green value.
                | blueStr:                      The desired blue value.
                '''
                # split data string
                [command, lightSourceName, redStr, greenStr, blueStr] = data.split(',')
                # recover RGB values
                red = float(redStr)
                green = float(greenStr)
                blue = float(blueStr)
                # update the light source's RGB values
                obj = scene.objects[lightSourceName]
                obj.color = [red, green, blue]
                print('changed color of light: %s to [%f,%f,%f]' % (lightSourceName, red, green, blue))
                # send control data
                controller['controlConnection'].send('AKN.'.encode('utf-8'))