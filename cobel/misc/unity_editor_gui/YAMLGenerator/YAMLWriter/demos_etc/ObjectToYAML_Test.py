from ObjectToYAML import ObjectToYAMLConverter as DB
from MathStuff import anglesToQuaternion

yamlDB = DB()
#YAML Tag
print(yamlDB.getYAML_TAG())
#Settings1
s1data = [1]
s1 = yamlDB.getYAML_OcclusionCullingSettings(s1data)
print(s1,end='')
#Settings2
s2data = [2]
s2 = yamlDB.getYAML_RenderSettings(s2data)
print(s2,end='')
#Settings3
s3data = [3]
s3 = yamlDB.getYAML_LightmapSettings(s3data)
print(s3,end='')
#Settings4
s4data = [4]
s4 = yamlDB.getYAML_NavMeshSettings(s4data)
print(s4,end='')
#GameObject
godata = [13,[14,15,17,18,19],"TestObj","untagged"]
go = yamlDB.getYAML_GameObject(godata)
print(go,end='')
#Transform
tdata = [14,13,anglesToQuaternion(67,50,1),[3,3,3],[1,1,1],[],0,0,[67,50,1]]
t = yamlDB.getYAML_Transform(tdata)
print(t,end='')
#BoxCollider
bcdata = [15,13,[2100000,"398476134",2],0,[1,1,1],[0,0,0]]
bc = yamlDB.getYAML_BoxCollider(bcdata)
print(bc,end='')
#MeshRenderer
mrdata = [17,13,[[4300000,"ct42378t",2],[4300000,"sudifb",2]]]
mr = yamlDB.getYAML_MeshRenderer(mrdata)
print(mr,end='')
#MeshFiler
mfdata = [18,13,[6900000,"ct4ddd",2]]
mf = yamlDB.getYAML_MeshFilter(mfdata)
print(mf,end='')
#CapsuleCollider
ccdata = [19,13,[],0,0.5,2,0,[1,2,3]]
cc = yamlDB.getYAML_CapsuleCollider(ccdata)
print(cc,end='')
#Camera
cdata = [20,32189,[1,1,1,0]]
c = yamlDB.getYAML_Camera(cdata)
print(c,end='')
#Light
ldata = [23,374,[1,0,0.3,1]]
l = yamlDB.getYAML_Light(ldata)
print(l,end='')
#Rigidbody
rbdata = [50,33334,1,0,0.3,0,1,1,80,1]
rb = yamlDB.getYAML_Rigidbody(rbdata)
print(rb,end='')
#MonoBehaviour
mbdata = [3245,21364,[11500000,"g7dsf7874nf",3],
		 [["emptyArr",[]],
		 ["onlyFileID",[13]],
		 ["someNumber",177013],
		 ["someString","hahLol"],
		 ["fullArr",[14,"f7nw48eh",2]]]]
mb = yamlDB.getYAML_MonoBehaviour(mbdata)
print(mb,end='')
#SphereCollider
scdata = [4232,145746,[4300000,"someGUIDLMAO",2],0,1,0.5,[0,1,2]]
sc = yamlDB.getYAML_SphereCollider(scdata)
print(sc,end='')
#MeshCollider
mcdata= [5234,1938456,[1274,"someID",3],0,1,30,[12742,"someID2",32]]
mc = yamlDB.getYAML_MeshCollider(mcdata)
print(mc,end='')