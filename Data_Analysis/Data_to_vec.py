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

Data = read_data("Event trigger Matrix of all channels.csv")
length = len(Data)

u = 1                                   #The dimension of half a scintillator
d = 1                                   #The thickness of the walls between scintillators
v = u + d/2                             #The coordinates of the first two scintillators next to the origin
w = 3*u + 3/2*d                         #The coordinates of the other two scintillators further out

print(length)

print(Data)

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

M = vec_to_matrix(np.array(Data.iloc[16]))

delta = event_to_vec(A)
r = np.sqrt(delta[0]**2+delta[1]**2+delta[2]**2)
    
theta_alt = np.arccos((-delta[2])/r)
print("Longitudinal angle corresponding to the vector:")
print(np.degrees(theta_alt))

print(M)

print("Vector corresponding to the event:")
print(event_to_vec(A))


######################################## Plotting the path ############################################

# Define side length of cube
a = 4*u + 3/2*d


# Define vertices of the cube
vertices = [(a, a, a), (a, a, -a), (a, -a, a), (a, -a, -a),
            (-a, a, a), (-a, a, -a), (-a, -a, a), (-a, -a, -a)]

# Define edges of the cube
edges = [(0,1), (0,2), (0,4),
         (1,3), (1,5),
         (2,3), (2,6),
         (4,5), (4,6),
         (7,5), (7,6), (7,3)]

# Define vector with direction (6,-6,-6)
v = event_to_vec(A)

# Define a point on the line
p0 = np.array([0, 0, 0])

# Create an array of points along the line
t = np.linspace(-4, 4, 10)
line_points = np.outer(t, v) + p0

# Create figure and axes objects
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot edges
for edge in edges:
    ax.plot(*zip(vertices[edge[0]], vertices[edge[1]]), color='black')

# Plot line
ax.plot(*line_points.T, color='red')

# Set limits and labels for axes
ax.set_xlim([-a-5, a+5])
ax.set_ylim([-a-5, a+5])
ax.set_zlim([-a-5, a+5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the viewing angle
ax.view_init(elev=20, azim=50)

# Show plot
plt.show()