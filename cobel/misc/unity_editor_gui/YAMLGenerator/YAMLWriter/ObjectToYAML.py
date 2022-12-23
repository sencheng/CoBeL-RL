import json

class ObjectToYAMLConverter:
	def __init__(self,dbPath='YAML_DB.json'): 
		global _db
		#load JSON-Database 
		with open(dbPath, 'r') as file:
			data = file.read()
		_db = json.loads(data)

	def getYAML_TAG(self):
		"""Returns a string, which is always at the start of an 
			UnityYAML-file.
		"""
		return _db["YAML_TAG"]

	def getYAML_GameObject(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,[componentID1,componentID2,...],Name,Tag]
		"""
		dbName = "GameObject"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1]
		if len(dataArr[1]) == 0:#components
			yaml += " []\n"
		else:
			yaml += "\n"
			for compID in dataArr[1]:
				yaml += _db[dbName][2] + str(compID) + _db[dbName][3]
		yaml += _db[dbName][4] + str(dataArr[2]) + _db[dbName][5] \
			 + str(dataArr[3]) + _db[dbName][6]
		return yaml

	def getYAML_Transform(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObjectID,[quaternionX,quaternionY,quaternionZ,quaternionW],
				  [x,y,z],[sX,sY,sZ],[childTransformID1,childTransformID2,...],fatherID,rootOrdner,[rX,rY,rZ]]
		rX,rY and rZ must be in degrees.
		"""
		dbName = "Transform"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		for i in range(4):#quaternion
			yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		for i in range(3):#position
			yaml += str(dataArr[3][i]) + _db[dbName][7+i]
		for i in range(3):#scale
			yaml += str(dataArr[4][i]) + _db[dbName][10+i]
		if len(dataArr[5]) == 0:#children
			yaml += " []\n"
		else:
			yaml += "\n"
			for childID in dataArr[5]:
				yaml += _db[dbName][13] + str(childID) + _db[dbName][14]
		yaml += _db[dbName][15] + str(dataArr[6]) + _db[dbName][16] \
			 + str(dataArr[7]) + _db[dbName][17]
		for i in range(3):#angles
			yaml += str(dataArr[8][i]) + _db[dbName][18+i]
		return yaml
		
	def getYAML_BoxCollider(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,[materialFileID,materialGUID,materialType]
				  isTrigger,enabled,[dX,dY,dZ],[centerX,centerY,centerZ]]

		if dataArr[2] (material) is empty, then the fileID will be 0.
		booleans are 0(false) or 1(true).
		"""
		dbName = "BoxCollider"#name inside the DB
		
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		if len(dataArr[2]) == 0:
			yaml += "0" + _db[dbName][5]
		else:
			for i in range(3):
				yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		yaml += str(dataArr[3]) + _db[dbName][6] + str(dataArr[4]) \
			 + _db[dbName][7]
		for i in range(3):
			yaml += str(dataArr[5][i]) + _db[dbName][8+i]
		for i in range(3):
			yaml += str(dataArr[6][i]) + _db[dbName][11+i]
		return yaml

	def getYAML_SphereCollider(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,[materialFileID,materialGUID,materialType]
				  isTrigger,enabled,radius,[centerX,centerY,centerZ]]

		if dataArr[2] (material) is empty, then the fileID will be 0.
		booleans are 0(false) or 1(true).
		"""
		dbName = "SphereCollider"#name inside the DB
		
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		if len(dataArr[2]) == 0:
			yaml += "0" + _db[dbName][5]
		else:
			for i in range(3):
				yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		yaml += str(dataArr[3]) + _db[dbName][6] + str(dataArr[4]) \
			 + _db[dbName][7] + str(dataArr[5]) + _db[dbName][8]
		for i in range(3):
			yaml += str(dataArr[6][i]) + _db[dbName][9+i]
		return yaml

	def getYAML_MeshCollider(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,[materialFileID,materialGUID,materialType]
				  isTrigger,convex,cookingOptions,[meshFileID,meshGUID,meshType]]

		if dataArr[2] or dataArr[6] (material) is empty, then the fileID will be 0.
		booleans are 0(false) or 1(true),
		"""
		dbName = "MeshCollider"#name inside the DB
		
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		if len(dataArr[2]) == 0:
			yaml += "0" + _db[dbName][5]
		else:
			for i in range(3):
				yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		yaml += str(dataArr[3]) + _db[dbName][6] + str(dataArr[4]) \
			 + _db[dbName][7] + str(dataArr[5]) + _db[dbName][8]
		if len(dataArr[6]) == 0:
			yaml += "0" + _db[dbName][11]
		else:
			for i in range(3):
				yaml += str(dataArr[6][i]) + _db[dbName][9+i]
		return yaml

	def getYAML_MeshRenderer(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,
				  [[mat1ID,mat1GUID,mat1Type],[mat2ID,mat2GUID,mat2Type],...]]

		for an extern material (file):
			matID: 2100000
			matGUID: the GUId of the file
			matType: 2
		for a built-in material:
			matID: the ID of the material
			matGUID: 0000000000000000f000000000000000
			matType: 0
		"""
		dbName = "MeshRenderer"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		if len(dataArr[2]) == 0:
			yaml += " []\n"
		else:
			for mat in dataArr[2]:
				for i in range(3):
					yaml += _db[dbName][3+i] + str(mat[i])
				yaml += _db[dbName][6]
		yaml += _db[dbName][7]
		return yaml

	def getYAML_MeshFilter(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,[materialFileID,materialGUID,materialType]]

		if dataArr[2] (material) is empty, then the fileID will be 0.
		for built-in material:
			matID: the ID of the material
			matGUID: 0000000000000000e000000000000000
			matType: 0
		"""
		dbName = "MeshFilter"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		if len(dataArr[2]) == 0:
			yaml += "0" + _db[dbName][5]
		else:
			for i in range(3):
				yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		return yaml

	def getYAML_OcclusionCullingSettings(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID]
		"""
		dbName = "OcclusionCullingSettings"#name inside the DB
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1]
		return yaml

	def getYAML_RenderSettings(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID]
		"""
		dbName = "RenderSettings"#name inside the DB
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1]
		return yaml

	def getYAML_LightmapSettings(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID]
		"""
		dbName = "LightmapSettings"#name inside the DB
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1]
		return yaml

	def getYAML_NavMeshSettings(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID]
		"""
		dbName = "NavMeshSettings"#name inside the DB
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1]
		return yaml

	def getYAML_CapsuleCollider(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,[materialFileID,materialGUID,materialType]
				  isTrigger,radius,height,direction,[centerX,centerY,centerZ]]

		if dataArr[2] (material) is empty, then the fileID will be 0.
		booleans are 0(false) or 1(true).
		direction must be:
			0->X-achsis
			1->Y-achsis
			2->Z-achsis
		"""
		dbName = "CapsuleCollider"#name inside the DB
		
		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		if len(dataArr[2]) == 0:
			yaml += "0" + _db[dbName][5]
		else:
			for i in range(3):
				yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		yaml += str(dataArr[3]) + _db[dbName][6] + str(dataArr[4]) \
			 + _db[dbName][7] + str(dataArr[5]) + _db[dbName][8] \
			 + str(dataArr[6]) + _db[dbName][9]
		for i in range(3):
			yaml += str(dataArr[7][i]) + _db[dbName][10+i]
		return yaml

	def getYAML_Camera(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,[bgcR,bgcG,bgcB,bgcA]]

		values of dataArr[2] (backgroundcolor) must be between 0 and 1.
		"""
		dbName = "Camera"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		for i in range(4):
			yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		return yaml

	def getYAML_Light(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,[colorR,colorG,colorB,colorA]]

		values of dataArr[2] (color) must be between 0 and 1.
		"""
		dbName = "Light"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		for i in range(4):
			yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		return yaml
		
	def getYAML_Rigidbody(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,mass,drag,angularDrag,useGravity,
				  isKinematic,interpolate,constraints,collisionDetection]

		constraints may be '80'.
		"""
		dbName = "Rigidbody"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		for i in range(8):
			yaml += str(dataArr[2+i]) + _db[dbName][3+i]
		return yaml

	def getYAML_MonoBehaviour(self,dataArr):
		"""Returns a YAML-String with the data from dataArr inserted.
		
		dataArr = [objectID,gameObject,
				  [scriptFileID,scriptGUID,scriptType],
				  [scriptParam1,scriptParam2,...]]

		scriptFileID may be 11500000
		scriptGUID may be the GUID of the script file
		scriptType may be 3
		scriptParameters must be a array of the format [paramName,paramValue]
			paramName must be a string
			paramValue may a number,string or an array.
			arrays have the format [fileID,GUID,type]
			-if array is empty, fileID will be 0 (missing parameter)
			-if GUID and type are missing, only fileID will be set (local parameter)
			-otherwise all 3 values must be given (extern parameter from other file)
		"""
		dbName = "MonoBehaviour"#name inside the DB

		yaml = _db[dbName][0] + str(dataArr[0]) + _db[dbName][1] \
			 + str(dataArr[1]) + _db[dbName][2]
		for i in range(3):
			yaml += str(dataArr[2][i]) + _db[dbName][3+i]
		for param in dataArr[3]:#scriptParameters
			yaml += "  " + str(param[0])#paramName
			if isinstance(param[1],list):#array
				if len(param[1]) == 0:#empty array
					yaml += _db[dbName][6] + "0" + _db[dbName][9]
				elif len(param[1]) == 1:#only fileID
					yaml += _db[dbName][6] + str(param[1][0]) + _db[dbName][9]
				else:#all values
					for i in range(3):
						yaml += _db[dbName][6+i] + str(param[1][i])
					yaml += _db[dbName][9]
			else:#number or string
				yaml += ": " + str(param[1]) +"\n"
		return yaml

	yamlFunctionMap = {
		"GameObject":getYAML_GameObject,
		"Transform":getYAML_Transform,
		"BoxCollider":getYAML_BoxCollider,
		"SphereCollider":getYAML_SphereCollider,
		"MeshCollider":getYAML_MeshCollider,
		"MeshRenderer":getYAML_MeshRenderer,
		"MeshFilter":getYAML_MeshFilter,
		"OcclusionCullingSettings":getYAML_OcclusionCullingSettings,
		"RenderSettings":getYAML_RenderSettings,
		"LightmapSettings":getYAML_LightmapSettings,
		"NavMeshSettings":getYAML_NavMeshSettings,
		"CapsuleCollider":getYAML_CapsuleCollider,
		"Camera":getYAML_Camera,
		"Light":getYAML_Light,
		"Rigidbody":getYAML_Rigidbody,
		"MonoBehaviour":getYAML_MonoBehaviour
	}

	def getYAML(self,yamlClassName,dataArr):
		"""Looks up the correct function to convert the
			object to YAML, uses it and returns the result.

		yamlClassName is the class name as a string.
		dataArr is the array, that shall be passed to the function.
		"""
		return self.yamlFunctionMap[yamlClassName](self,dataArr)

	def getYAML_FullScene(self,yamlObjArr):
		"""Generates a YAML-String that that can be written to a file.
			This file will then be readable by Unity.

		yamlObjArr is an array of yamlObjects. It must have the format:
		[[yamlClassName1,dataArr1],[yamlClassName2,dataArr2],...]
			yamlClassName is the name of the class of the object as a string.
			dataArr is the data, that will be inserted for the object
				in the YAML-file. 
			You can find all supported Class Names in yamlFunctionMap.
			You can check the documentations of the functions getYAML_<ClassName>()
				to get information on the dataArr's for objects
		"""
		yaml = self.getYAML_TAG()
		for yamlObj in yamlObjArr:
			yaml += self.getYAML(yamlObj[0],yamlObj[1])
		return yaml