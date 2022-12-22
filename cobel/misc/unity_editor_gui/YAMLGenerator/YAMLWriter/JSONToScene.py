from ObjectToYAML import ObjectToYAMLConverter as DB
from DataToObjectArray import generateObjectArray
import json
import sys

def generateScene(jsonString):
	jsonObject = json.loads(jsonString)
	#extract object data from json
	planeSettings = [[jsonObject["width"],jsonObject["height"]],
		[jsonObject["texture"]["fileID"],jsonObject["texture"]["guid"],jsonObject["texture"]["type"]]]
	#GUIs y-achsis is flipped, have to calculate for all y-coordinates:
	#newY = planeHeight-oldY
	planeHeight = planeSettings[0][1]
	if jsonObject["reward"] is None:
		rewardPosition = None
	else:
		rewardPosition = [jsonObject["reward"]["posX"],
			planeHeight-jsonObject["reward"]["posY"]]
	if jsonObject["agent"] is None:
		agentSettings = None
	else:
		agentSettings = [
			[
				jsonObject["agent"]["posX"],
				planeHeight-jsonObject["agent"]["posY"]
			],0]#agent rotation from json?
	walls = []
	for wall in jsonObject["walls"]:
		walls.append([
				[wall["posX"],planeHeight-wall["posY"]],
				[wall["posX_end"],planeHeight-wall["posY_end"]],
				[wall["texture"]["fileID"],wall["texture"]["guid"],wall["texture"]["type"]],
				[jsonObject["wallWidth"],jsonObject["wallHeight"]]
			])
	#expecting 32-bit number from java 0xAARRGGBB
	lightA = ((jsonObject["lightColor"]&0xFF000000)>>24)/255
	lightR = ((jsonObject["lightColor"]&0xFF0000)>>16)/255
	lightG = ((jsonObject["lightColor"]&0xFF00)>>8)/255
	lightB = (jsonObject["lightColor"]&0xFF)/255
	lightColor = [lightR,lightG,lightB,lightA]

	#generate ObjectArray
	objectArray = generateObjectArray(planeSettings,
		rewardPosition,agentSettings,
		walls,lightColor)
	#write and return YAMLString
	db = DB()
	yamlString = db.getYAML_FullScene(objectArray)
	return yamlString

if __name__ == "__main__":
	"""Usage: 
		JSONToScene [jsonFile] <outputFileName>
		jsonFile can be the name of the json file or 
			the json itself
		if outputFileName is not given, 
			the result will instead be printed
	"""
	if sys.argv[1].endswith(".json"):
		with open(sys.argv[1], 'r') as file:
			data = file.read()
		yamlString = generateScene(data)
	else:
		yamlString = generateScene(sys.argv[1])

	if len(sys.argv) > 2:
		with open(sys.argv[2],"w") as file:
			file.write(yamlString);
			#print message/info to GUI on success?
	else:
		print(yamlString)