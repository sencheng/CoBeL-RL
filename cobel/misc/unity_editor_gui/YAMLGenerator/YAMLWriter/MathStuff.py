from numpy import sin,cos,deg2rad,arctan2,pi#float32

def anglesToQuaternion(rX,rY,rZ):
	"""Converts an euler angle to a YXZ quaternion.

	rX,rY,rZ must be in degrees.
	"""
	cX = cos(deg2rad(rX/2))
	cY = cos(deg2rad(rY/2))
	cZ = cos(deg2rad(rZ/2))
	sX = sin(deg2rad(rX/2))
	sY = sin(deg2rad(rY/2))
	sZ = sin(deg2rad(rZ/2))
	qW = cX*cY*cZ + sX*sY*sZ
	qX = sX*cY*cZ + cX*sY*sZ
	qY = cX*sY*cZ - sX*cY*sZ
	qZ = cX*cY*sZ - sX*sY*cZ
	return [qX,qY,qZ,qW]

def getVerticesFromEdges(edges):
	"""Returns an array of the vertices of an array of edges.

	edges = [[[x1,y1],[x2,y2]],
			[[x3,y3],[x4,y4]],...]
	"""
	vertices=[]
	for edge in edges:
		for vertex in edge:
			if not (vertex in vertices):
				vertices.append(vertex)
	return vertices

def getVerticesConnectedToVertex(vertex, allEdges):
	"""Returns an array of all vertices, that are connected to a
		specified vertex through an edge inside a given set of edges.

	vertex = [x,y]
	allEdges = [[[x,y],[x2,y2]],
 			   [[x3,y3],[x,y]],...]
	"""
	connectedVertices=[]
	for edge in allEdges:
		#if one point of the edge is the same as the vertex, 
		#then the other point is connected to the vertex
		for i in range(len(edge)):
			if edge[i] == vertex:
				#only two vertices, use other vertex
				connectedVertices.append(edge[(i+1)%2])
	return connectedVertices

def getAngleBetween(vector1, vector2):
	"""Returns the angle of vector2 relative to vector1, 
		when going counter-clockwise from vector1.

	vector = [x,y]
	The angle is normalized to the range [0,2pi)
	"""
	angle = arctan2(vector2[1],vector2[0])-arctan2(vector1[1],vector1[0])
	#normalize angle to [0,2pi)
	if angle < 0:
		angle += 2*pi
	return angle

#https://stackoverflow.com/questions/31169125/outermost-polygon-from-a-set-of-edges
def getOuterPolygonOfEdges(edges):
	"""Calculates and returns the outer polygon defined by the edges.
	An edge must not appear twice in edges.
	All edges must be either inside of the polygon or they 
		must be an outer edge of the polygon (no backtracking).

	Returns array of vertices. The vertices are sorted in the order,
		in which they connect through the edges. The first vertex
		connects to the last vertex.
	Returns None, if a outer polygon could not be built.
		An outer polygon can not be generated, 
		if there are less than 2 vertices given or, 
		if the edges don't connect properly.

	edges = [[[x1,y1],[x2,y2]],[[x2,y2],[x3,y3]],...]
	"""
	polygon = []
	vertices = getVerticesFromEdges(edges)
	if len(vertices) < 2:
		return None

	#get left most vertex (use the lowest one of those)
	currentVertex = vertices[0]
	for vertex in vertices[1:]:
		if vertex[0] < currentVertex[0]:
			currentVertex = vertex
		elif vertex[0] == currentVertex[0]:
			if vertex[1] < currentVertex[1]:
				currentVertex = vertex
	#initialise polygon so that vector1 goes downwards at the start
	#first vertex will be removed, since it is only used to initialise the algorithm
	polygon.append([currentVertex[0],currentVertex[1]-1])
	polygon.append(currentVertex)

	while True:#do-while
		lastlastVertex = polygon[len(polygon)-2]
		lastVertex = polygon[len(polygon)-1]

		connectedVertices = getVerticesConnectedToVertex(lastVertex,edges)
		#do not go backwards
		if lastlastVertex in connectedVertices:
			connectedVertices.remove(lastlastVertex)
		#if there is only one connected next vertex,
		#you can just pick this one and continue
		if len(connectedVertices) == 1:
			#if the first (real) vertex is encountered again, break
			if polygon[1] == connectedVertices[0]:
				break
			polygon.append(connectedVertices[0])
			continue
		elif len(connectedVertices) == 0:
			#there is no way to connect to another vertex
			return None
		vector1 = [lastlastVertex[0]-lastVertex[0],lastlastVertex[1]-lastVertex[1]]
		#calculate the next vertex
		nextVertexWithAngle = None
		for vertex in connectedVertices:
			vector2 = [vertex[0]-lastVertex[0],vertex[1]-lastVertex[1]]
			#angle of vector2 relative to vector1, 
			#when going counter-clockwise from vector1
			angle = getAngleBetween(vector1,vector2)
			if nextVertexWithAngle is None:
				nextVertexWithAngle = [vertex,angle]
				continue
			if angle < nextVertexWithAngle[1]:
				nextVertexWithAngle = [vertex,angle]

		#if the first (real) vertex is encountered again, break
		if polygon[1] == nextVertexWithAngle[0]:
			break
		polygon.append(nextVertexWithAngle[0])

	return polygon[1:]