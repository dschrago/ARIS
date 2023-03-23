import smbus
import math
import time
import csv
import datetime
import numpy as np

class MPU:
    def __init__(self, gyro, acc, tau):
        # Class / object / constructor setup
        self.gx = None; self.gy = None; self.gz = None;
        self.ax = None; self.ay = None; self.az = None;

        self.gyroXcal = 0
        self.gyroYcal = 0
        self.gyroZcal = 0

        self.gyroRoll = 0   
        self.gyroPitch = 0
        self.gyroYaw = 0

        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        self.dtTimer = 0
        self.tau = tau

        self.gyroScaleFactor, self.gyroHex = self.gyroSensitivity(gyro)
        self.accScaleFactor, self.accHex = self.accelerometerSensitivity(acc)

        self.bus = smbus.SMBus(1)
        self.address = 0x68

    def gyroSensitivity(self, x):
        # Create dictionary with standard value of 500 deg/s
        return {
            250:  [131.0, 0x00],
            500:  [65.5,  0x08],
            1000: [32.8,  0x10],
            2000: [16.4,  0x18]
        }.get(x,  [65.5,  0x08])

    def accelerometerSensitivity(self, x):
        # Create dictionary with standard value of 4 g
        return {
            2:  [16384.0, 0x00],
            4:  [8192.0,  0x08],
            8:  [4096.0,  0x10],
            16: [2048.0,  0x18]
        }.get(x,[8192.0,  0x08])

    def setUp(self):
        # Activate the MPU-6050
        self.bus.write_byte_data(self.address, 0x6B, 0x00)

        # Configure the accelerometer
        self.bus.write_byte_data(self.address, 0x1C, self.accHex)

        # Configure the gyro
        self.bus.write_byte_data(self.address, 0x1B, self.gyroHex)

        # Display message to user
        print("MPU set up:")
        print('\tAccelerometer: ' + str(self.accHex) + ' ' + str(self.accScaleFactor))
        print('\tGyro: ' + str(self.gyroHex) + ' ' + str(self.gyroScaleFactor) + "\n")
        time.sleep(2)

    def eightBit2sixteenBit(self, reg):
        # Reads high and low 8 bit values and shifts them into 16 bit
        h = self.bus.read_byte_data(self.address, reg)
        l = self.bus.read_byte_data(self.address, reg+1)
        val = (h << 8) + l

        # Make 16 bit unsigned value to signed value (0 to 65535) to (-32768 to +32767)
        if (val >= 0x8000):
            return -((65535 - val) + 1)
        else:
            return val

    def getRawData(self):
        self.gx = self.eightBit2sixteenBit(0x43)
        self.gy = self.eightBit2sixteenBit(0x45)
        self.gz = self.eightBit2sixteenBit(0x47)

        self.ax = self.eightBit2sixteenBit(0x3B)
        self.ay = self.eightBit2sixteenBit(0x3D)
        self.az = self.eightBit2sixteenBit(0x3F)

    def calibrateGyro(self, N):
        # Display message
        print("Calibrating gyro with " + str(N) + " points. Do not move!")

        # Take N readings for each coordinate and add to itself
        for ii in range(N):
            self.getRawData()
            self.gyroXcal += self.gx
            self.gyroYcal += self.gy
            self.gyroZcal += self.gz

        # Find average offset value
        self.gyroXcal /= N
        self.gyroYcal /= N
        self.gyroZcal /= N

        # Display message and restart timer for comp filter
        print("Calibration complete")
        print("\tX axis offset: " + str(round(self.gyroXcal,3)))
        print("\tY axis offset: " + str(round(self.gyroYcal,3)))
        print("\tZ axis offset: " + str(round(self.gyroZcal,3)) + "\n")
        time.sleep(2)
        self.dtTimer = time.time()

    def processIMUvalues(self):
        # Update the raw data
        self.getRawData()

        # Subtract the offset calibration values
        self.gx -= self.gyroXcal
        self.gy -= self.gyroYcal
        self.gz -= self.gyroZcal

        # Convert to instantaneous degrees per second
        self.gx /= self.gyroScaleFactor
        self.gy /= self.gyroScaleFactor
        self.gz /= self.gyroScaleFactor

        # Convert to g force
        self.ax /= self.accScaleFactor
        self.ay /= self.accScaleFactor
        self.az /= self.accScaleFactor

    def compFilter(self):
        # Get the processed values from IMU
        self.processIMUvalues()

        # Get delta time and record time for next call
        dt = time.time() - self.dtTimer
        self.dtTimer = time.time()

        # Acceleration vector angle
        accPitch = math.degrees(math.atan2(self.ay, self.az))
        accRoll = math.degrees(math.atan2(self.ax, self.az))

        # Gyro integration angle [in degrees not radians]
        self.gyroRoll -= self.gy * dt
        self.gyroPitch += self.gx * dt
        self.gyroYaw += self.gz * dt
        self.yaw = self.gyroYaw

        # Comp filter
        self.roll = (self.tau)*(self.roll - self.gy*dt) + (1-self.tau)*(accRoll)
        self.pitch = (self.tau)*(self.pitch + self.gx*dt) + (1-self.tau)*(accPitch)


        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

        # Print data
        print("Time:{:.4f}\tAccel_x:{:.4f}\tAccel_y:{:.4f}\tAccel_z:{:.4f}\tRoll:{:.4f}\tPitch:{:.4f}\tYaw:{:.4f} ".format(timestamp,round(str(self.ax),3), round(str(self.ay),3), round(str(self.az),3), round(str(self.roll,3)), round(str(self.pitch,3)), round(str(self.yaw,3)) ))
        
        # Return values
        #return {'t':timestamp, 'x': round(self.ax,3), 'y': round(self.ay,3), 'z': round(self.az,3), 'roll' : round(self.roll,3), 'pitch' : round(self.pitch,3),'yaw' : round(self.yaw,3) } 


def main():
    # Set up class
    gyro = 250      # 250, 500, 1000, 2000 [deg/s]  here we want 2000 deg/s
    acc = 2         # 2, 4, 7, 16 [g]   here we want +/- 16 g
    tau = 0.98
    mpu = MPU(gyro, acc, tau)

    # Set up sensor and calibrate gyro with N points
    mpu.setUp()
    mpu.calibrateGyro(500)

    # Initialize list for data in format: Time ; Accel_x [g] ;  Accel_y [g] ;  Accel_z [y] ; Roll [deg] ;  Pitch [deg] ;  Yaw [deg]   
    #data = []
    
    # Run for 20 secounds
    startTime = time.time()
    while(time.time() < (startTime + 20)):
        # Gather data and print in terminal (at the end take print out)
        mpu.compFilter()
        # Append data to the list
        #all_data = mpu.compFilter()
        #np.vstack([data, [all_data['t'],all_data['x'], all_data['y'], all_data['z'], all_data['roll'], all_data['pitch'], all_data['yaw']]])
        #print("Time:{:.4f}\tAccel_x:{:.4f}\tAccel_y:{:.4f}\tAccel_z:{:.4f}\tRoll:{:.4f}\tPitch:{:.4f}\tYaw:{:.4f} ".format(all_data['t'],all_data['x'], all_data['y'], all_data['z'], all_data['roll'], all_data['pitch'], all_data['yaw'] ))
        


    # End
    print("Closing")

     # Convert list to numpy array/matrix
    #data_matrix = np.array(data)
    
    # Save matrix to CSV file
    #np.savetxt("data.csv", data_matrix, delimiter=",", header="Time,Accel_x,Accel_y,Accel_z,Roll,Pitch,Yaw")

    # Print confirmation message
    #print("Data saved to data.csv")

# Main loop
if __name__ == '__main__':
	main()
