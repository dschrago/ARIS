import numpy as np
import matplotlib.pyplot as plt
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import random
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def read_data(filename):
    """
    Reads data from a CSV file and returns a pandas DataFrame.
    """
    data = pd.read_csv(filename)
    data.drop(["34_16", "34_17"], axis="columns", inplace=True)  # drop two columns
    return data

Data = read_data("Event trigger Matrix of all channels.csv")
length = len(Data)
#length = 50000

u = 0.6                                 #The dimension of half a scintillator side in [cm]
d = 1/3                                 #The thickness of the walls between scintillators in [cm]
a = u + d/2                             #The coordinates of the first two scintillators next to the origin in [cm]
b = 3*u + 3/2*d                         #The coordinates of the other two scintillators further out in [cm]

 
err_x = u                               #Error in x-direction
err_y = u                               #Error in y-direction
err_z = u                               #Error in z-direction

print("Event number:")
print(length)

#print("Measurement Data")
#print(Data)

def filter(vec):
    """
    Checks if an event is valid so if there are no two triggers seperated by more than one
    
    Args:
    vec: numpy array of shape (4,)
    
    Returns:
    True or False
    """
    for i in range(np.size(vec)-2):
        if ((vec[i] != 0) and (vec[i+2] != 0)):
            return False
        elif ((i==0) and (vec[i]!=0) and vec[i+3]!=0):
            return False
        else: 
            continue
    return True

def neighbor(vec):
    """
    Checks if there are neighbours in the vector so if it is for example as follows: [0,1,1,0]

    Returns:
    True or False
    """
    for i in range(np.size(vec)-1):
        if((vec[i] == 1) and (vec[i+1] == 1)):
            return True
        
        else: continue
    
    return False

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

def get_pos_x(vec):
    """
    Get the x position of a scintillator given its trigger vector.
    
    Args:
    vec: numpy array of shape (4,), the trigger vector
    
    Returns:
    float, the x position of the scintillator
    """
    x = 0 
        
    if neighbor(vec):
        if ((vec[0] == 1) and (vec[1] == 1)):
            x = (b + a)/2
        if ((vec[1] == 1) and (vec[2] == 1)):
            x = 0
        if ((vec[2] == 1) and (vec[3] == 1)):
            x = (-b - a)/2

    else:
        if (vec[0] == 1):
            x = b
        if (vec[1] == 1):
            x = a
        if (vec[2] == 1):
            x = -a
        if (vec[3] == 1):
            x = -b

    return x

def get_pos_y(vec):
    """
    Get the y position of a scintillator given its trigger vector.
    
    Args:
    vec: numpy array of shape (4,), the trigger vector
    
    Returns:
    float, the y position of the scintillator
    """
    y = 0

    if neighbor(vec):
        if ((vec[0] == 1) and (vec[1] == 1)):
            y = (-b - a)/2
        if ((vec[1] == 1) and (vec[2] == 1)):
            y = 0
        if ((vec[2] == 1) and (vec[3] == 1)):
            y = (b + a)/2

    else:
        if (vec[0] == 1):
            y = -b
        elif (vec[1] == 1):
            y = -a
        elif (vec[2] == 1):
            y = a
        elif (vec[3] == 1):
            y = b
        
    return y

def azimuth(vec):
    """
    Calculates the azimuthal angle of our vector

    Returns:
    Angle 
    """
    x = vec[0]
    y = vec[1]
    return np.arctan2(y,x)
    
def has_event(vec):
    """
    Returns True if there was any event at all, False otherwise.
    """
    return any(n > 0 for n in vec)

def offset(coords):
    """
    Returns an offset for the starting point in x-direction of the first layer for a more correct plot. 
    The offset is determined based on where the rods were last hit in the last layer 
    """
    delta_x = coords[3][0] - coords[1][0]
    if (coords[1][0] != coords[3][0]):
        if (coords[3][0] == -b):
            return 3*u
        elif (coords[3][0] == -a):
            return u
        elif (coords[3][0] == a):
            return -u
        elif (coords[3][0] == b):
            return -3*u
        
        else:
            return 0

    else:
        return 0

def err_theta(x,y,z):
    """
    Calculates the error of the longitudinal angle we calculated based on gaussian error propagation and assuming that the error is +- u for x,y,z since we assume
    the coordinates to be in the middle of the rod but an offset of u in either direction for x,y,z would be plausible.

    Returns:
    Error
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if (r == np.abs(z)):
        return 0
    else:
        delta_theta = np.sqrt(((-x*z)/(r**3 * np.sqrt(1 - (z**2/(r**2))))*err_x)**2 + ((-y*z)/(r**3 * np.sqrt(1 - (z**2/(r**2))))*err_y)**2 + ((x**2 + y**2)/(r**3 * np.sqrt(1 - (z**2/(r**2))))*err_z)**2)
        return delta_theta

def get_rotated_vector(Rm, vector):
    """
    Function to rotate vector based on the rotation matrix that was calculated for each event using the MPU and taking the closest 
    possible timestamp between MPU and WD.
    
    Returns:
    Rotated vector for angle calculation.
    """
    vec_rotated = np.dot(Rm, vector)
    return vec_rotated
    
def err_phi(x,y):
    delta_phi = np.sqrt((-(y)/(x**2 + y**2)*err_x)**2 + (x/(x**2 + y**2)*err_y)**2)
    return delta_phi



#Old version of the function
'''
def event_to_vec(M):
    """
    Converts a matrix of all channel triggers of one event into a spatial vector. We only consider events where all four layers
    are hit by a muon.
    """
    layers = []
    mistake = []

    delta_x_13 = 0 
    delta_x_24 = 0
    delta_y_13 = 0
    delta_y_24 = 0
    delta_z_13 = 0
    delta_z_24 = 0


    for i in range(4):
        #First checks if there even was an event
        if has_event(M[i,:]):
            #Then checks is the layers i have valid events

            if ((i == 0) and filter(M[i,:])):
                x_1 = 0
                y_1 = get_pos_y(M[i,:])
                z_1 = w
                layers.append(1)

            elif ((i == 0) and not filter(M[i,:])):
                x_1 = 0 
                y_1 = 0
                z_1 = 0
                mistake.append(1)
                print("Invalid input in layer 1")
                continue
            
            if ((i == 1) and filter(M[i,:])):
                x_2 = get_pos_x(M[i,:])
                y_2 = 0
                z_2 = v
                layers.append(2)
            
            elif ((i == 1) and not filter(M[i,:])):
                x_2 = 0 
                y_2 = 0
                z_2 = 0
                mistake.append(2)
                print("Invalid input in layer 2")
                continue
                
            
            if ((i == 2) and filter(M[i,:])):
                x_3 = 0
                y_3 = get_pos_y(M[i,:])
                z_3 = -v
                layers.append(3)
            
            elif ((i == 2) and not filter(M[i,:])):
                x_3 = 0
                y_3 = 0
                z_3 = 0
                mistake.append(3)
                print("Invalid input in layer 3")
                continue
                

            if ((i == 3) and filter(M[i,:])):
                x_4 = get_pos_x(M[i,:])
                y_4 = 0
                z_4 = -w
                layers.append(4)
            
            elif ((i == 3) and not filter(M[i,:])):
                x_4 = 0
                y_4 = 0
                z_4 = 0
                mistake.append(4)
                print("Invalid input in layer 4")
                continue
                
    #Checks first if the layers are empty which would mean no event so if true there is at least one trigger on one layer
    if bool(layers):
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
        
        if 1 in mistake or 3 in mistake:
            delta_x_13 = 0
            delta_y_13 = 0
            delta_z_13 = 0
        
        if 2 in mistake or 4 in mistake:
            delta_x_24 = 0
            delta_y_24 = 0
            delta_z_24 = 0

        #Vector
        delta = np.array([delta_x_24, delta_y_13, -(2*d + 4* u)])

        #Starting position for the plot
        pos = np.array([x_2, y_1, z_1])

        #Angle
        r = np.sqrt(delta[0]**2+delta[1]**2+delta[2]**2)
        if (r == 0):
            angle = 0
        else:
            angle = np.arccos((-delta[2])/r)

        return delta, pos, angle

    else:
        print("There was an invalid event in one of the layers.")
        return [0,0,0], [0,0,0], 0
'''
#New version of the function
def event_to_vec(M):
    """
    Converts a matrix of all channel triggers of one event into a spatial vector. We only consider events where all four layers
    are hit by a muon. It checks if the events are valid and then calculates the longitudinal angle.

    Returns:
    The vector delta of the muon path, the starting position pos for the plot, the longitudinal angle and the error of the calculation
    """

    layers = []
    mistake = []

    coords = {0: (0, get_pos_y(M[0, :]), b),
              1: (get_pos_x(M[1, :]), 0, a),
              2: (0, get_pos_y(M[2, :]), -a),
              3: (get_pos_x(M[3, :]), 0, -b)}
    
    #Rotation matrix from MPU
    #rot_angle = random.uniform(0,np.pi)
    rot_angle = 0
    RM = [[1,0,0],
              [0,np.cos(rot_angle),-np.sin(rot_angle)],
              [0,np.sin(rot_angle),np.cos(rot_angle)]]
    
    #Rotated coordinates 
    coords_rot = {0: get_rotated_vector(RM,(0, get_pos_y(M[0, :]), b)),
              1: get_rotated_vector(RM,(get_pos_x(M[1, :]), 0, a)),
              2: get_rotated_vector(RM,(0, get_pos_y(M[2, :]), -a)),
              3: get_rotated_vector(RM,(get_pos_x(M[3, :]), 0, -b))}

    for i in range(4):
        if has_event(M[i, :]):
            if filter(M[i, :]):
                x, y, z = coords_rot[i]
                layers.append(i + 1)
            else:
                mistake.append(i + 1)
                print(f"Invalid input in layer {i + 1}")

    if len(layers) == 4:
        delta_x_13 = 0
        delta_y_13 = coords_rot[2][1] - coords_rot[0][1]
        delta_z_13 = coords_rot[2][2] - coords_rot[0][2]
        delta_13 = np.array([delta_x_13, delta_y_13, delta_z_13])
        

        delta_x_24 = coords_rot[3][0] - coords_rot[1][0]
        delta_y_24 = 0
        delta_z_24 = coords_rot[3][2] - coords_rot[1][2]
        delta_24 = np.array([delta_x_24, delta_y_24, delta_z_24])

        if any(layer in mistake for layer in [1, 3]):
            delta_x_13 = 0
            delta_y_13 = 0
            delta_z_13 = 0

        if any(layer in mistake for layer in [2, 4]):
            delta_x_24 = 0
            delta_y_24 = 0
            delta_z_24 = 0

        #Vector
        #Correction : delta = np.array([delta_x_24, delta_y_13, -(2 * d + 4 * u)])
        delta = np.array([delta_x_24, delta_y_13, delta_z_13])

        #Starting position for the plot with correction in x-direction for the plot
        pos = np.array([coords_rot[1][0] + offset(coords_rot), coords_rot[0][1], coords_rot[0][2]])

        #Angle
        r = np.sqrt(delta[0]**2 + delta[1]**2 + delta[2]**2)
        if (r == 0):
            angle = 0
            error = 0
        else:
            angle = np.arccos((-delta[2])/r)
            error = err_theta(delta[0],delta[1],delta[2])
            if(angle > np.pi/2):
                angle = np.abs(angle - np.pi)
            
        
        #rot_delta = get_rotated_vector(RM,delta)

        return delta, pos, angle, error

    else:
        print("There was an invalid event in one of the layers.")
        return [0,0,0], [0,0,0], 0

angles = []
azi = []
err = []

#Simulation of 500 random events
def simulation():
    """
    Simulates amount of events corresponding to length by generating a random matrix to then calculate the angles and errors.

    Returns:
    Nothing 
    """
    for j in range(length):

        A = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

        for i in range(4):
            index_1 = random.randint(0,3)
            index_2 = random.randint(0,3)
            A[i,index_1] = 1

            if ((index_1 == index_2 - 1) or (index_1 == index_2 + 1)):
                A[i,index_2] = 1
            elif (index_1 == index_2):
                A[i,index_2] = 1
            else:
                A[i,index_2] = 0

        if (np.sum(A) == 0):
            print("There was an invalid event in one of the layers.")
        
        else:
            delta_i, pos_i, angle_i, error_i = event_to_vec(A)
            azi_angle = azimuth(delta_i)
            angles.append(angle_i)
            azi.append(azi_angle)
            err.append(error_i)
    
    angles_av = np.degrees(np.average(angles))
    azi_av = np.degrees(np.average(azi))
    err_av = np.degrees(np.average(err))


    print(f"Average longitudinal angle of {length} events:")
    print(angles_av)
    print(f"Average azimuthal angle of {length} events:")
    print(azi_av)
    print(f"Average error of {length} events:")
    print(err_av)
    print(f"Average relative Error of {length} events:")
    print(err_av/angles_av)

simulation()

'''
#Go through all data to transform into matrix and calculate the corresponding spatial vector
for i in range(length):
    angles_theta = []
    errors = []
    angles_phi = []

    B = vec_to_matrix(np.array(Data.iloc[i]))
    
    if (np.sum(B) == 0):
        print("There was an invalid event in one of the layers.")
    
    else:
        delta, pos, theta_alt, error_th  = event_to_vec(B)
        phi = azimuth(delta)
        angles_phi.append
        angles_theta.append(theta_alt)
        errors.append(error_th)

'''

M = vec_to_matrix(np.array(Data.iloc[16]))

B = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

for i in range(4):
    index_1 = random.randint(0,3)
    index_2 = random.randint(0,3)
    B[i,index_1] = 1

    if ((index_1 == index_2 - 1) or (index_1 == index_2 + 1)):
        B[i,index_2] = 1
    elif (index_1 == index_2):
        B[i,index_2] = 1
    else:
        B[i,index_2] = 0

print(B)
delta, pos, theta_alt, error_th = event_to_vec(B)

theta_alt = np.degrees(theta_alt)           #Converting to degrees
azimuth_alt = np.degrees(azimuth(delta))
error_th = np.degrees(error_th)

print("Longitudinal angle corresponding to the vector:")
print(theta_alt)

print("Azimuthal angle cooresponding to the vector:")
print(azimuth_alt)

#print(M)

print("Vector corresponding to the event:")
print(delta)

print("Error of the calculation")
print(error_th)

print("Relative Error of the calculation:")
print(np.degrees(error_th)/(np.degrees(theta_alt)))

print(f"Total angle: {theta_alt} +- {error_th}")

######################################## Plotting the path ############################################

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

    #print(np.array(vertices))
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
ve = event_to_vec(B)[0]

# Define a point on the line
p0 = pos

# Create an array of points along the line
t = np.linspace(-4, 4, 10)
line_points = np.outer(t, ve) + p0

# Create figure and axes objects

fig = plt.figure()
plt.subplot(111, projection='3d')

plt.gca().add_collection3d(Poly3DCollection(vec1, alpha=.05, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec2, alpha=.05, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec3, alpha=.05, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec4, alpha=.05, edgecolor='k', facecolor='red'))

plt.gca().add_collection3d(Poly3DCollection(vec5, alpha=.05, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec6, alpha=.05, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec7, alpha=.05, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec8, alpha=.05, edgecolor='k', facecolor='teal'))

plt.gca().add_collection3d(Poly3DCollection(vec9, alpha=.05, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec10, alpha=.05, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec11, alpha=.05, edgecolor='k', facecolor='red'))
plt.gca().add_collection3d(Poly3DCollection(vec12, alpha=.05, edgecolor='k', facecolor='red'))

plt.gca().add_collection3d(Poly3DCollection(vec13, alpha=.05, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec14, alpha=.05, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec15, alpha=.05, edgecolor='k', facecolor='teal'))
plt.gca().add_collection3d(Poly3DCollection(vec16, alpha=.05, edgecolor='k', facecolor='teal'))


ax = fig.gca()
  
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
ax.view_init(elev=20, azim=50)

plt.show()

#Plot of the longitudinal angles and errors
event = np.linspace(0,length, length)
plt.figure()
plt.errorbar(event, np.degrees(angles),yerr=np.degrees(err),fmt='.')
plt.xlabel("Event number")
plt.ylabel("Longitudinal angle of the event [$^{\circ}$]")

#Plot of the histogramm of the longitudinal angles
plt.figure()
plt.hist(np.degrees(angles),10)

plt.figure()
plt.plot(event, np.degrees(azi),'.')
plt.xlabel("Event number")
plt.ylabel("Azimuthal angle of the event [$^{\circ}$]")

plt.figure()
plt.hist(np.degrees(azi),20)

# Show plot
#plt.show(block=True)