#draw hexagonal grid
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

xDomain=np.linspace(-1,1,9)
yDomain=np.linspace(-1,1,9)

period = xDomain[1] - xDomain[0]
grid_points = np.array([[x, y] for y in yDomain for x in xDomain])
shift_indices =  []

for y in yDomain[1::2] :
    shift_indices.append(np.where(grid_points[:,1]==y)[0])

s = np.ravel(shift_indices)
    
for idx in s:
    grid_points[idx][0] += period/2 

del_indices = np.where(grid_points[:,0] > 1)[0]
grid_points = np.delete(grid_points,del_indices,0)

mesh = Delaunay(grid_points,qhull_options='Qt Qbb Qc')
mesh_points = mesh.points
mesh_elements = mesh.simplices

ne = mesh_elements.shape[0]
edges = np.array([mesh_elements[:,0], mesh_elements[:,1], 
                    mesh_elements[:,1], mesh_elements[:,2],
                    mesh_elements[:,2], mesh_elements[:,0]]).T.reshape(3*ne,2)
edges = np.sort(edges)

edges = np.unique(edges,axis=0)

fig, ax = plt.subplots()

for e in edges : 
    ax.plot(mesh_points[e,0],mesh_points[e,1],c='black',alpha=0.5)

ax.scatter(mesh_points[:,0],mesh_points[:,1],c='blue')

ax.set_aspect("equal") 
ax.axis("off")
