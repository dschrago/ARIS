import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

def read_data(filename):
    """
    Reads data from a CSV file and returns a pandas DataFrame.
    """
    data = pd.read_csv(filename)
    data.drop(["34_16", "34_17"], axis="columns", inplace=True)  # drop two columns
    return data

#Data = read_data("Event trigger Matrix of all channels.csv")
#length = len(Data)

u = 1                                   #The dimension of half a scintillator
d = 1                                   #The thickness of the walls between scintillators
v = u + d/2                             #The coordinates of the first two scintillators next to the origin
w = 3*u + 3/2*d                         #The coordinates of the other two scintillators further out

#print(length)

#print(Data)

def vec_to_matrix(vec):
    """
    Convert a vector of length 16 into a 4x4 matrix.
    
    Args:
    vec: numpy array of shape (16,), the vector to convert
    
    Returns:
    numpy array of shape (4, 4), the resulting matrix
    """
    matrix = np.reshape(vec, (4, 4))
    return matrix

def get_pos_y(vec):
    """
    Get the y position of a scintillator given its trigger vector.
    
    Args:
    vec: numpy array of shape (4,), the trigger vector
    
    Returns:
    float, the y position of the scintillator
    """
    y = 0
    if (vec[0] == 1):
        y = -w
    elif (vec[1] == 1):
        y = -v
    elif (vec[2] == 1):
        y = v
    elif (vec[3] == 1):
        y = w
        
    return y
    
def get_pos_x(vec):
    """
    Get the x position of a scintillator given its trigger vector.
    
    Args:
    vec: numpy array of shape (4,), the trigger vector
    
    Returns:
    float, the x position of the scintillator
    """
    x = 0 
    if (vec[0] == 1):
        x = w
    elif (vec[1] == 1):
        x = v
    elif (vec[2] == 1):
        x = -v
    elif (vec[3] == 1):
        x = -w
        
    return x
    
def has_event(vec):
    """
    Returns True if there was any event at all, False otherwise.
    """
    return any(n > 0 for n in vec)


def event_to_vec(M):
    """
    Converts a matrix of all channel triggers of one event into a spatial vector.
    """
    layers = []

    for i in range(4):
        if has_event(M[i,:]):
            if i == 0:
                x_1 = 0
                y_1 = get_pos_y(M[i,:])
                z_1 = w
                layers.append(1)
            elif i == 1:
                x_2 = get_pos_x(M[i,:])
                y_2 = 0
                z_2 = v
                layers.append(2)
            elif i == 2:
                x_3 = 0
                y_3 = get_pos_y(M[i,:])
                z_3 = -v
                layers.append(3)
            elif i == 3:
                x_4 = get_pos_x(M[i,:])
                y_4 = 0
                z_4 = -w
                layers.append(4)

    if 1 in layers and 3 in layers:
        delta_x_13 = 0
        delta_y_13 = y_3 - y_1
        delta_z_13 = z_3 - z_1
        delta_13 = np.array([delta_x_13, delta_y_13, delta_z_13])
    
    if 2 in layers and 4 in layers:
        delta_x_24 = x_4 - x_2
        delta_y_24 = 0
        delta_z_24 = z_4 - z_2
        delta_24 = np.array([delta_x_24, delta_y_24, delta_z_24])

    delta = np.array([delta_x_24, delta_y_13, delta_z_13])

    return delta

    
A = np.array([[1,0,0,0],[0,0,0,1],[1,0,0,0],[1,0,0,0]])

#M = vec_to_matrix(np.array(Data.iloc[16]))

delta = event_to_vec(A)
r = np.sqrt(delta[0]**2+delta[1]**2+delta[2]**2)
    
theta_alt = np.arccos((-delta[2])/r)
print("Longitudinal angle corresponding to the vector:")
print(np.degrees(theta_alt))

#print(M)

print("Vector corresponding to the event:")
print(event_to_vec(A))


######################################## Plotting the path ############################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#input values

d = 1/3 # gap between blocks
v = 1.2 #length of side of cuboid
l = 5.8 # depth of cuboid

# x coordinate: start with right point back, to right point front, to left front to left back 
# y coordinate: start with right point back, to right point front, to left front to left back 
# z coordinate: start with bottom point, to top point


############################ LAYER 1 ############################

# Block1
x1 = [-l/2, l/2, l/2, -l/2]
y1 = [-v - 3/2*d ,-v - 3/2*d, -2*v - 3/2*d,-2*v - 3/2*d]    
z1 = [3/2*d+v,3/2*d+2*v]

# Block2
x2 = [-l/2, l/2, l/2, -l/2]
y2 = [-d/2 ,-d/2, -v -d/2,-v - d/2]    
z2 = [3/2*d+v,3/2*d+2*v]

# Block3
x3 = [-l/2, l/2, l/2, -l/2]
y3 = [d/2+v ,d/2+v, d/2,d/2]    
z3 = [3/2*d+v,3/2*d+2*v]

# Block4
x4 = [-l/2, l/2, l/2, -l/2]
y4 = [3/2*d+2*v ,3/2*d+2*v, 3/2*d+v,3/2*d+v]    
z4 = [3/2*d+v,3/2*d+2*v]


############################ LAYER 2 ############################

# Block5
x5 = [2*v + 3/2*d ,v+3/2*d, v+3/2*d,2*v+3/2*d]    
y5 = [l/2, l/2, -l/2, -l/2]  
z5 = [d/2,d/2+v]

# Block6
x6 = [d/2 ,v+d/2, v+d/2,d/2]    
y6 = [l/2, l/2, -l/2, -l/2]  
z6 = [d/2,d/2+v]

# Block7
x7 = [-v -d/2 ,-d/2, -d/2,-v-d/2]    
y7 = [l/2, l/2, -l/2, -l/2]  
z7 = [d/2,d/2+v]

# Block8
x8 = [-2*v-3/2*d ,-v-3/2*d, -v-3/2*d,-2*v-3/2*d]    
y8 = [l/2, l/2, -l/2, -l/2]  
z8 = [d/2,d/2+v]


############################ LAYER 3 ############################

# Block9
x9 = [-l/2, l/2, l/2, -l/2]
y9 = [-v - 3/2*d ,-v - 3/2*d, -2*v - 3/2*d,-2*v - 3/2*d]    
z9 = [-d/2-v,-d/2]

# Block10
x10 = [-l/2, l/2, l/2, -l/2]
y10 = [-d/2 ,-d/2, -v -d/2,-v - d/2]    
z10 = [-d/2-v,-d/2]

# Block11
x11 = [-l/2, l/2, l/2, -l/2]
y11 = [d/2+v ,d/2+v, d/2,d/2]    
z11 = [-d/2-v,-d/2]

# Block12
x12 = [-l/2, l/2, l/2, -l/2]
y12 = [3/2*d+2*v ,3/2*d+2*v, 3/2*d+v,3/2*d+v]    
z12 = [-d/2-v,-d/2]


############################ LAYER 4 ############################

# Block13
x13 = [2*v + 3/2*d ,v+3/2*d, v+3/2*d,2*v+3/2*d]    
y13= [l/2, l/2, -l/2, -l/2]  
z13= [-3/2*d-2*v,-3/2*d-v]

# Block14
x14 = [d/2 ,v+d/2, v+d/2,d/2]    
y14 = [l/2, l/2, -l/2, -l/2]  
z14 = [-3/2*d-2*v,-3/2*d-v]

# Block15
x15 = [-v -d/2 ,-d/2, -d/2,-v-d/2]    
y15 = [l/2, l/2, -l/2, -l/2]  
z15 = [-3/2*d-2*v,-3/2*d-v]

# Block16
x16 = [-2*v-3/2*d ,-v-3/2*d, -v-3/2*d,-2*v-3/2*d]    
y16 = [l/2, l/2, -l/2, -l/2]  
z16 = [-3/2*d-2*v,-3/2*d-v]

def edgecoord(pointx, pointy, pointz):
    edgex = [pointx[0], pointx[1], pointx[1], pointx[0]]
    edgey = [pointy[0], pointy[1], pointy[1], pointy[0]]
    edgez = [pointz[0], pointz[0], pointz[1], pointz[1]]
    return list(zip(edgex, edgey, edgez))

def coordConvert(x, y, lheight, uheight):  # if orientation is wrong, exchange y and x
    if len(x) != len(y) and len(x)>2:
        return
    vertices=[]
    #Top layer
    vertices.append(list(zip(x, y, list(np.full(len(x), uheight)))))
    # Side layers
    for it in np.arange(len(x)):
        it1 = it + 1
        if it1 >= len(x):
            it1 = 0
        vertices.append(edgecoord([x[it], x[it1]], [y[it], y[it1]], [lheight, uheight]))
    #Bottom layer
    vertices.append(list(zip(x, y, list(np.full(len(x), lheight)))))

    print(np.array(vertices))
    return vertices


# Blocks
vec1 = coordConvert(x1, y1, z1[0], z1[1])
vec2 = coordConvert(x2, y2, z2[0], z2[1])
vec3 = coordConvert(x3, y3, z3[0], z3[1])
vec4 = coordConvert(x4, y4, z4[0], z4[1])

vec5 = coordConvert(x5, y5, z5[0], z5[1])
vec6 = coordConvert(x6, y6, z6[0], z6[1])
vec7 = coordConvert(x7, y7, z7[0], z7[1])
vec8 = coordConvert(x8, y8, z8[0], z8[1])

vec9 = coordConvert(x9, y9, z9[0], z9[1])
vec10 = coordConvert(x10, y10, z10[0], z10[1])
vec11 = coordConvert(x11, y11, z11[0], z11[1])
vec12 = coordConvert(x12, y12, z12[0], z12[1])

vec13 = coordConvert(x13, y13, z13[0], z13[1])
vec14= coordConvert(x14, y14, z14[0], z14[1])
vec15= coordConvert(x15, y15, z15[0], z15[1])
vec16= coordConvert(x16, y16, z15[0], z16[1])


# Define vector with direction (6,-6,-6)
ve = event_to_vec(A)

# Define a point on the line
p0 = np.array([0, 0, 0])

# Create an array of points along the line
t = np.linspace(-4, 4, 10)
line_points = np.outer(t, ve) + p0

# Create figure and axes objects
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fig = plt.figure()
plt.subplot(111, projection='3d')

plt.gca().add_collection3d(Poly3DCollection(vec1, alpha=.1, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec2, alpha=.1, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec3, alpha=.1, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec4, alpha=.1, edgecolor='k', facecolor='red'))

plt.gca().add_collection3d(Poly3DCollection(vec5, alpha=.1, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec6, alpha=.1, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec7, alpha=.1, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec8, alpha=.1, edgecolor='k', facecolor='teal'))

plt.gca().add_collection3d(Poly3DCollection(vec9, alpha=.1, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec10, alpha=.1, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec11, alpha=.1, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec12, alpha=.1, edgecolor='k', facecolor='red'))

plt.gca().add_collection3d(Poly3DCollection(vec13, alpha=.1, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec14, alpha=.1, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec15, alpha=.1, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec16, alpha=.1, edgecolor='k', facecolor='teal'))


ax = fig.gca(projection='3d')
  
# Change the orientation of the axes
ax.set_xlim([-4, 4])
ax.set_ylim([-4, 4])
ax.set_zlim([-4, 4])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')


# Plot line
ax.plot(*line_points.T, color='red')

# Set limits and labels for axes

# Set the viewing angle
ax.view_init(elev=35, azim=15)

# Show plot
plt.show()