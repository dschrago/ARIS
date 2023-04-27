import numpy as np
import csv
import bisect


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
    
    ######################### MPU DATA ############################

    # CSV file with MPU data to be read out
    filename = "MPU.csv"  # filename has to be adjusted

    # Open MPU CSV file
    with open(filename, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        data = []
        for row in csv_reader:
            data.append(row)
    
    # Time list with relative time differences
    time_data = [sub_list[0] for sub_list in data]
    time_mpudata = [(float(time_str) - float(time_data[0])) for time_str in time_data]  # Calculate the relative time differences
    #time_mpudata[0] = 0  # Set the first entry to 0

    print(time_mpudata)
    
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

    
    ######################### WD TIME DATA ############################

    # CSV file with WD time data to be read out
    filename = "Timestamps.csv"  # filename has to be adjusted

    # Open WD TIME CSV file
    with open(filename, mode="r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader)
        data = []
        for row in csv_reader:
            data.append(row)
    
    # Time list with relative time differences
    time_data = [sub_list[0] for sub_list in data]
    time_wddata = [(float(time_str) - float(time_data[0])) for time_str in time_data]  # Calculate the relative time differences
    #time_wddata[0] = 0  # Set the first entry to 0


    ###################### Time pairing ######################

    # Find closest match in time_mpudata for each value in time_wddata
    
    mpu_indices = [bisect.bisect_left(time_mpudata, time_val) for time_val in time_wddata]

    print(mpu_indices)  

    rotation_matrices = []
    for idx in mpu_indices:
        R = get_rotation_matrix(yaw_data[idx], pitch_data[idx], roll_data[idx])
        rotation_matrices.append(R)

    # Convert the rotation matrices list into a csv file
    with open("rotation_matrices.csv", mode="w") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Rotation matrix'])
        for R in rotation_matrices:
            csv_writer.writerow([R])


# Main loop
if __name__ == '__main__':
	main()
