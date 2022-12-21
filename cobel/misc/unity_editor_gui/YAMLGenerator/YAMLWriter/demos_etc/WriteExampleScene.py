from ObjectToYAML import ObjectToYAMLConverter as DB
from DataToObjectArray import generateObjectArray
from MathStuff import anglesToQuaternion

testArr = [["OcclusionCullingSettings",[1]],["RenderSettings",[2]],["LightmapSettings",[3]],["NavMeshSettings",[4]],
		  ["GameObject",[5,[6,7,8,9],"Cylinder (1)","Untagged"]],
		  ["Transform",[6,5,anglesToQuaternion(0,0,0),[4.1,8.47,-6.66],[4,2,4],[],0,0,[0,0,0]]],
		  ["CapsuleCollider",[7,5,[],0,0.5,2,1,[0,0,0]]],
		  ["MeshRenderer",[8,5,[[2100000,"60b2f363763cdedaeab35b0fc5217878",2]]]],
		  ["MeshFilter",[9,5,[10206,"0000000000000000e000000000000000",0]]]]

testjsondata = [[[9,10],[2100000,"cff0f8a4e9c85a045be6d610d20b87b3",2]],
	[2,2],
	[[0.5,0.5],45],
	[
		[[0,0],[9,0],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[0,0],[0,6],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[9,0],[9,10],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[0,2],[2,2],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[2,2],[2,4],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[2,4],[7,4],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[5,0],[5,2],[2100000,"69fefdd39d2b34b169e921910bed9c0d",2]],
		[[7,4],[7,6],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],

		#[[0,7],[5,10],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		#[[5,6],[0,10],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],

		[[5,6],[5,10],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[0,6],[5,6],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[9,8],[7,8],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[9,10],[5,10],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]],
		[[5,2],[7,2],[2100000,"2e687a6242afc4a7e993fec6042518a9",2]]
	],
	[1,244/255,214/255,0]
]
testArr2 = generateObjectArray(testjsondata[0],testjsondata[1],
	testjsondata[2],testjsondata[3],testjsondata[4])

db = DB()
scene = db.getYAML_FullScene(testArr)
scene2 = db.getYAML_FullScene(testArr2)
print(scene2)
with open("testScene.unity","w") as file:
	file.write(scene2);