import numpy as np
import csv


# Get Rotation Matrix
def get_rotation_matrix(yaw, pitch, roll):
    
    # Yaw Matrix
    yawMatrix = np.matrix([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
        ])

    # Pitch Matrix
    pitchMatrix = np.matrix([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
        ])

    # Roll Matrix
    rollMatrix = np.matrix([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
        ])
        
    # Rotation Matrix
    R = yawMatrix * pitchMatrix * rollMatrix
    return R


# Get rotated vector for given Rotation Matrix and Vector
def get_rotated_vector(Rm, vector):
    vec_rotated = np.dot(Rm, vector)
    return vec_rotated


def main():
    
    # CSV file with MPU data to be read out
    filename = "03-29_15:03:03.033082.csv"  # filename has to be adjusted

    # Open CSV file
    with open(filename, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        data = []
        for row in csv_reader:
            data.append(row)
            
    # Print data
    #print(data)
    
    # Time list
    time_data = [sub_list[0] for sub_list in data]
    
    # Yaw list
    yaw_data = [sub_list[6] for sub_list in data]
    yaw_data = [np.deg2rad(float(i)) for i in yaw_data] #convert to radians
    #print(yaw_data)
    
    # Pitch list
    pitch_data = [sub_list[5] for sub_list in data]
    pitch_data = [np.deg2rad(float(i)) for i in pitch_data] #convert to radians
    #print(pitch_data)
    
    # Roll list
    roll_data = [sub_list[4] for sub_list in data]
    roll_data = [np.deg2rad(float(i)) for i in roll_data] #convert to radians
    #print(roll_data)
    
    
    # CSV file with vector data to be read out
    #filename = "vector.csv"  # filename has to be adjusted

    # Open CSV file
    #with open(filename, mode="r") as csv_file:
        #csv_reader = csv.reader(csv_file, delimiter=",")
        #next(csv_reader) #skips first row
        #vector = []
        #for row in csv_reader:
            #vector.append(row)
    
    # List of Vectors obtained from WD data showing trajectory of muon
    # print(vector)  
    
    # To be checked: vector needs to be in the from that vector[0] = np.array([x,y,z])
            
    
    # Initialize an empty list to store the rotation matrices
    Rotation_matrices = []
    
    # Loop through the elements of the yaw, pitch and roll data lists
    for i in range(len(yaw_data)):
        rot_mat = get_rotation_matrix(yaw_data[i], pitch_data[i], roll_data[i])
        Rotation_matrices.append(rot_mat)
    
    # Initialize an empty list to store the rotation matrices
    Rot_Vec = []
    
    # Get rotated vectors in matrix format
    for i in range(len(Rotation_matrices)):
        vec_rot = get_rotated_vector(Rotation_matrices[i], vector[i])
        Rot_Vec.append(vec_rot)
    
    

# Main loop
if __name__ == '__main__':
	main()
