from MathStuff import anglesToQuaternion,getOuterPolygonOfEdges
from numpy import rad2deg,arctan2,sqrt

def nextObjectID():
    """Counter for generating an unique objectID 
        for each object.
    """
    nextObjectID.lastObjectID += 1
    return nextObjectID.lastObjectID
nextObjectID.lastObjectID = 0

def generatePlane(planeSettings):
    """Generates the plane object and returns
        an array of the components of the plane.

    planeSettings = [[planeWidth,planeHeight],[matFileID,matGUID,matType]]
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Plane","floor"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(0,0,0),
        [planeSettings[0][0]/2,0,planeSettings[0][1]/2],
        [planeSettings[0][0]/10,1,planeSettings[0][1]/10],
        [],0,0,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshFilter = ["MeshFilter",[nextObjectID(),gameObject[1][0],
        generatePlane.planeMesh]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshRenderer = ["MeshRenderer",[nextObjectID(),gameObject[1][0],
        [planeSettings[1]]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshCollider = ["MeshCollider",[nextObjectID(),gameObject[1][0],
        [],0,0,30,generatePlane.planeMesh]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    planeComponents = [gameObject,transform,meshFilter,meshRenderer,meshCollider]
    return planeComponents
generatePlane.planeMesh = ["10209","0000000000000000e000000000000000",0]

def generateReward(rewardPosition):
    """Generates the reward object and returns
        an array of the components of the reward.

    rewardPosition = [x,y]
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Reward","Untagged"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(0,0,0),
        [rewardPosition[0],0.3,rewardPosition[1]],
        [0.8,0.8,0.8],
        [],0,0,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshFilter = ["MeshFilter",[nextObjectID(),gameObject[1][0],
        generateReward.sphereMesh]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshRenderer = ["MeshRenderer",[nextObjectID(),gameObject[1][0],
        [generateReward.greenMaterial]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    sphereCollider = ["SphereCollider",[nextObjectID(),
        gameObject[1][0],[],0,0,0.5,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)
    
    rewardComponents = [gameObject,transform,meshFilter,meshRenderer,sphereCollider]
    return rewardComponents
generateReward.sphereMesh = ["10207","0000000000000000e000000000000000",0]
generateReward.greenMaterial = ["2100000","c67450f290f3e4897bc40276a619e78d",2]

def generateEventSystem():
    """Generates the eventsystem object and returns
        an array of the components of the eventsystem.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"EventSystem","Untagged"]]

    transform = ["Transform",[nextObjectID(),
        gameObject[1][0],
        anglesToQuaternion(0,0,0),[0,0,0],[1,1,1],[],0,0,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    eventSystemMonoBehaviour = ["MonoBehaviour",[nextObjectID(),gameObject[1][0],
        generateEventSystem.eventSystemScript,
        [["m_FirstSelected",[]],["m_sendNavigationEvents",1],["m_DragThreshold",5]]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    simMonoBehaviour = ["MonoBehaviour",[nextObjectID(),gameObject[1][0],
        generateEventSystem.standaloneInputModuleScript,
        [["m_HorizontalAxis","Horizontal"],["m_VerticalAxis","Vertical"],
            ["m_SubmitButton","Submit"],["m_CancelButton","Cancel"],
            ["m_InputActionsPerSecond",10],["m_RepeatDelay",0.5],["m_ForceModuleActive",0]]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)
    
    eventSystemComponents = [gameObject,transform,eventSystemMonoBehaviour,simMonoBehaviour]
    return eventSystemComponents
generateEventSystem.eventSystemScript = ["11500000","76c392e42b5098c458856cdf6ecaaaa1",3]
generateEventSystem.standaloneInputModuleScript = ["11500000","4f231c4fb786f3946a6b90b886c48677",3]

def generateDirectionalLight(lightColor):
    """Generates the directional light object and returns
        an array of the components of the directional light.

    lightColor = [r,g,b,a]
        All values of lightcolor must be between 0 and 1.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Directional Light","Untagged"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(90,0,0),
        [0,5,0],
        [1,1,1],
        [],0,0,[90,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    light = ["Light",[nextObjectID(),gameObject[1][0],lightColor]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    directionalLightComponents = [gameObject,transform,light]
    return directionalLightComponents

def generateMainCamera():
    """Generates the main camera object and returns an
        array of the components of the main camera.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Main Camera","MainCamera"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(90,0,0),
        [0,14,-3],
        [1,1,1],
        [],0,0,[90,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    camera = ["Camera",[nextObjectID(),gameObject[1][0],[49/255,77/255,121/255,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    mainCameraComponents = [gameObject,transform,camera]
    return mainCameraComponents

def generateAgentCamera(fatherID):
    """Generates the agent camera object and returns an
        array of the components of the agent camera.

    fatherID is the ID of the transform of the parent object.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"AgentCamera","Untagged"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(0,0,0),
        [0,0.3,0.1],
        [1,1,1],
        [],fatherID,0,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    camera = ["Camera",[nextObjectID(),gameObject[1][0],[56/255,56/255,56/255,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)
    
    agentCameraComponents = [gameObject,transform,camera]
    return agentCameraComponents

def generateAgentCylinder(fatherID):
    """Generates the agent cylinder object and returns
        an array of the components of the agent cylinder.

    fatherID is the ID of the transform of the parent object.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Cylinder","Untagged"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(90,0,0),
        [0,0.2,0.46],
        [0.5,0.05,0.5],
        [],fatherID,0,[90,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshFilter = ["MeshFilter",[nextObjectID(),gameObject[1][0],
        generateAgentCylinder.cylinderMesh]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshRenderer = ["MeshRenderer",[nextObjectID(),gameObject[1][0],
        [generateAgentCylinder.defaultMaterial]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    capsuleCollider = ["CapsuleCollider",[nextObjectID(),gameObject[1][0],[],0,0.5,2,1,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)
    
    agentCylinderComponents = [gameObject,transform,meshFilter,meshRenderer,capsuleCollider]
    return agentCylinderComponents
generateAgentCylinder.cylinderMesh = ["10206","0000000000000000e000000000000000",0]
generateAgentCylinder.defaultMaterial = ["10303","0000000000000000f000000000000000",0]

def generateAgent(agentSettings,planeID,mazeID,perimeterNodesID):
    """Generates the agent object and returns an array
        of the components of the agent and its children.

    agentSettings = [[x,y],rotation]
        rotation must be in degrees.
    planeID,mazeID and perimeterNodesID are the IDs 
        of the gameObject components of the objects.

    This function calls generateAgentCamera() and generateAgentCylinder()
        to generate the children.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Agent","Untagged"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(0,agentSettings[1],0),
        [agentSettings[0][0],0.2,agentSettings[0][1]],
        [0.6,0.4,0.6],
        [],0,0,[0,agentSettings[1],0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    agentCameraComponents = generateAgentCamera(transform[1][0])
    transform[1][5].append(agentCameraComponents[1][1][0])

    agentCylinderComponents = generateAgentCylinder(transform[1][0])
    transform[1][5].append(agentCylinderComponents[1][1][0])

    meshFilter = ["MeshFilter",[nextObjectID(),gameObject[1][0],
        generateAgent.cubeMesh]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshRenderer = ["MeshRenderer",[nextObjectID(),gameObject[1][0],
        [generateAgent.agentBlueMaterial]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    boxCollider = ["BoxCollider",[nextObjectID(),gameObject[1][0],[],0,1,[1,1,1],[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    rigidbody = ["Rigidbody",[nextObjectID(),gameObject[1][0],10,4,0.1,1,0,0,80,0]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    behaviourParameters = ["MonoBehaviour",[nextObjectID(),gameObject[1][0],
        generateAgent.behaviourParametersScript,
        [["m_BrainParameters","\n    VectorObservationSize: 0\n"+
        "    NumStackedVectorObservations: 1\n    m_ActionSpec:\n"+
        "      m_NumContinuousActions: 3\n      BranchSizes: \n"+
        "    VectorActionSize: 03000000\n    VectorActionDescriptions: []\n"+
        "    VectorActionSpaceType: 1\n    hasUpgradedBrainParametersWithActionSpec: 1"],
            ["m_Model",[]],["m_InferenceDevice",0],["m_BehaviorType",0],
            ["m_BehaviorName","TAgent"],["TeamId",0],["m_UseChildSensors",1],
            ["m_UseChildActuators",1],["m_ObservableAttributeHandling",0]]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    cameraSensor = ["MonoBehaviour",[nextObjectID(),gameObject[1][0],
        generateAgent.cameraSensorScript,
        [["m_Camera",[agentCameraComponents[2][1][0]]],["m_SensorName","CameraSensor"],["m_Width",84],
            ["m_Height",84],["m_Grayscale",0],["m_ObservationStacks",1],["m_Compression",1]]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    generalAgent = ["MonoBehaviour",[nextObjectID(),gameObject[1][0],
        generateAgent.generalAgentScript,
        [["agentParameters","\n    maxStep: 0"],["hasUpgradedFromAgentParameters",1],
            ["MaxStep",100000],["floor",[planeID]],["maze",[mazeID]],
            ["perimeter",[perimeterNodesID]],["timeBetweenDecisionsAtInference",0]]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)
    
    agentComponents = [gameObject,transform,meshFilter,meshRenderer,boxCollider,
        rigidbody,behaviourParameters,cameraSensor,generalAgent]
    agentComponents += agentCylinderComponents + agentCameraComponents
    return agentComponents
generateAgent.cubeMesh = ["10202","0000000000000000e000000000000000",0]
generateAgent.agentBlueMaterial = ["2100000","c9fa44c2c3f8ce74ca39a3355ea42631",2]
generateAgent.behaviourParametersScript = ["11500000","5d1c4e0b1822b495aa52bc52839ecb30",3]
generateAgent.cameraSensorScript = ["11500000","282f342c2ab144bf38be65d4d0c4e07d",3]
generateAgent.generalAgentScript = ["11500000","3f263207fb7c8cb7eaa23404119958e8",3]

def generateWall(wall,fatherID):
    """Generates a wall object and returns an
        array of the components of the wall.

    wall = [[x1,y1],[x2,y2],[materialFileID,materialGUID,materialType],
           [wallWidth,wallHeight]]
    fatherID is the ID of the transform of the parent object.

    """
    wallCenter = [(wall[0][0]+wall[1][0])/2, (wall[0][1]+wall[1][1])/2]
    wallAngle = -rad2deg(arctan2(wall[1][1]-wallCenter[1], wall[1][0]-wallCenter[0]))
    wallLength = sqrt((wall[1][0]-wall[0][0])**2 + (wall[1][1]-wall[0][1])**2)

    gameObject = ["GameObject",[nextObjectID(),[],"Wall","wall"]]
    
    #wall will be a cube, rotated around the y-achsis.
    #the x-scale is the length of the wall
    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(0,wallAngle,0),
        [wallCenter[0],wall[3][1]/2,wallCenter[1]],
        [wallLength+wall[3][0],wall[3][1],wall[3][0]],
        [],fatherID,0,[0,wallAngle,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshFilter = ["MeshFilter",[nextObjectID(),gameObject[1][0],
        generateWall.cubeMesh]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshRenderer = ["MeshRenderer",[nextObjectID(),
        gameObject[1][0],[wall[2]]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    boxCollider = ["BoxCollider",[nextObjectID(),gameObject[1][0],[],0,1,[1,1,1],[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)
    
    wall = [gameObject,transform,meshFilter,meshRenderer,boxCollider]
    return wall
generateWall.cubeMesh = ["10202","0000000000000000e000000000000000",0]

def generateWalls(walls):
    """Generates the maze container object and returns an array
        of the components of the wall container and its children.

    walls = [[[x1,y1],[x2,y2],[matFileID,matGUID,matType],[wallWidth,wallHeight]],
            [[x3,y3],[x2,y2],[matFileID,matGUID,matType],[wallWidth,wallHeight]],
            ...]

    This function calls generateWall() to generate the walls.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Maze","Untagged"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(0,0,0),
        [0,0,0],
        [1,1,1],
        [],0,0,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    wallsComponents = [gameObject,transform]
    for wall in walls:
        wallComponents = generateWall(wall,transform[1][0])
        transform[1][5].append(wallComponents[1][1][0])
        wallsComponents += wallComponents

    return wallsComponents

def generatePerimeterNode(position,fatherID):
    """Generates a perimeter node object and returns an
        array of the components of the perimeter node.

    position = [x,y]
    fatherID is the ID of the transform of the parent object.
    """
    gameObject = ["GameObject",[nextObjectID(),[],"Perimeter node","Untagged"]]

    transform = ["Transform",[nextObjectID(),
        gameObject[1][0],anglesToQuaternion(0,0,0),
        [position[0],0,position[1]],
        [0.1,0.1,0.1],
        [],fatherID,0,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshFilter = ["MeshFilter",[nextObjectID(),gameObject[1][0],
        generatePerimeterNode.cubeMesh]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    meshRenderer = ["MeshRenderer",[nextObjectID(),gameObject[1][0],
        [generatePerimeterNode.blackMaterial]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    boxCollider = ["BoxCollider",[nextObjectID(),gameObject[1][0],[],0,0,[1,1,1],[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    perimeterNodeComponents = [gameObject,transform,meshFilter,meshRenderer,boxCollider]
    return perimeterNodeComponents
generatePerimeterNode.cubeMesh = ["10202","0000000000000000e000000000000000",0]
generatePerimeterNode.blackMaterial = ["2100000","69fefdd39d2b34b169e921910bed9c0d",2]

def generatePerimeterNodes(walls,planeDimensions):
    """Generates the perimeter node container and 
        returns an array of the components of the 
        perimeter node container and its children.

    Perimeter nodes are generated at the outer most 
        corners of all walls. If the outer walls
        do not make up a simple polygon, then 4 
        perimeter nodes are generated at the corners 
        of the plane instead.

    walls = [[[x1,y1],[x2,y2]],
            [[x3,y3],[x4,y4]],...]
    planeDimensions = [planeWidth,planeHeight]

    This function calls generatePerimeterNode() to generate
        the perimeter nodes.
    """
    positions = getOuterPolygonOfEdges(walls) 

    gameObject = ["GameObject",[nextObjectID(),[],"Perimeter nodes","Untagged"]]

    transform = ["Transform",[nextObjectID(),gameObject[1][0],
        anglesToQuaternion(0,0,0),
        [0,0,0],
        [1,1,1],
        [],0,0,[0,0,0]]]
    gameObject[1][1].append(nextObjectID.lastObjectID)

    nodesComponents = [gameObject,transform]
    if positions is None:#If no polygon could be generated
        #generate 4 perimeter nodes at the corners of the plane
        for y in range(2):
            for x in range(2):
                nodeComponents = generatePerimeterNode([x*planeDimensions[0],y*planeDimensions[1]],transform[1][0])
                transform[1][5].append(nodeComponents[1][1][0])
                nodesComponents += nodeComponents
    else:
        #else generate perimeter nodes at the corners of the polygon
        for position in positions:
            nodeComponents = generatePerimeterNode(position,transform[1][0])
            transform[1][5].append(nodeComponents[1][1][0])
            nodesComponents += nodeComponents
    
    return nodesComponents
   
def generateObjectArray(planeSettings,rewardPosition,agentSettings,walls,lightColor):
    """Generates all objects in a scene and returns
        an array of all components of all objects.

    planeSettings = [[planeWidth,planeHeight],[matFileID,matGUID,matType]]
    rewardPosition = [x,y]
    agentSettings = [[x,y],rotation]
        rotation must be in degrees.
    walls = [[[x1,y1],[x2,y2],[matFileID,matGUID,matType],[wallWidth,wallHeight]],
            [[x1,y1],[x2,y2],[matFileID,matGUID,matType],[wallWidth,wallHeight]],
            ...]
    lightColor = [r,g,b,a]
        All values of lightColor must be between 0 and 1.
    """
    objectArray = [["OcclusionCullingSettings",[nextObjectID()]],
        ["RenderSettings",[nextObjectID()]],
        ["LightmapSettings",[nextObjectID()]],
        ["NavMeshSettings",[nextObjectID()]]]

    mainCamera = generateMainCamera()
    objectArray += mainCamera

    light = generateDirectionalLight(lightColor)
    objectArray += light

    eventSystem = generateEventSystem()
    objectArray += eventSystem

    if rewardPosition is not None:
        reward = generateReward(rewardPosition)
        objectArray += reward

    plane = generatePlane(planeSettings)
    objectArray += plane

    maze = generateWalls(walls)
    objectArray += maze

    #strip unnecessery information from the walls for perimeter nodes
    wallEdges = []
    for wall in walls:
        wallEdges.append([wall[0],wall[1]])

    perimeterNodes = generatePerimeterNodes(wallEdges,planeSettings[0])
    objectArray += perimeterNodes

    if agentSettings is not None:
        agent = generateAgent(agentSettings,
            plane[0][1][0],maze[0][1][0],perimeterNodes[0][1][0])
        objectArray += agent

    return objectArray