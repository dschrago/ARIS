import smbus
import math
import time
import csv
import datetime
import numpy as np
import RPi.GPIO as GPIO

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

        self.dtTimer = 0     # self.tau term is a smoothing factor, which determines the contribution of the gyroscope and accelerometer readings to the final angle calculation.
        self.tau = tau       # A higher self.tau value will place more weight on the accelerometer readings, while a lower self.tau value will place more weight on the gyroscope readings.
   
        self.gyroScaleFactor, self.gyroHex = self.gyroSensitivity(gyro)
        self.accScaleFactor, self.accHex = self.accelerometerSensitivity(acc)

        self.bus = smbus.SMBus(1)
        self.address = 0x68

    def gyroSensitivity(self, x):
        # Create dictionary with standard value of 2000 deg/s
        return {
            250:  [131.0, 0x00],
            500:  [65.5,  0x08],
            1000: [32.8,  0x10],
            2000: [16.4,  0x18]
        }.get(x,  [65.5,  0x08])

    def accelerometerSensitivity(self, x):
        # Create dictionary with standard value of 16 g
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
        #print(" Time: " + str(timestamp) \
        #    + " Roll: " + str(round(self.roll,3)) \
        #    + " Pitch: " + str(round(self.pitch,3)) \
        #    + " Yaw: " + str(round(self.yaw,3)) \
        #    + " Accel_x: " + str(round(self.ax,3)) \
        #    + " Accel_y: " + str(round(self.ay,3)) \
        #    + " Accel_z: " + str(round(self.az,3)) )
        
        # Return values
        return timestamp, round(self.ax,3),  round(self.ay,3),  round(self.az,3),  round(self.roll,3),  round(self.pitch,3), round(self.yaw,3) 

# Relay on
def relay_on(pin):
    GPIO.output(pin, GPIO.HIGH)

# Relay off
def relay_off(pin):
    GPIO.output(pin, GPIO.LOW)
    

def main():
    # Set up class
    gyro = 2000      # 250, 500, 1000, 2000 [deg/s]    ; for rocket we want 2000 deg/s  
    acc = 16         # 2, 4, 8, 16 [g]                 ; for rocket we want +/- 16 g    
    tau = 1
    mpu = MPU(gyro, acc, tau)
    
    # Variable for counter and Relay
    relay_state = GPIO.LOW

    # Set up sensor and calibrate gyro with N points
    mpu.setUp()
    mpu.calibrateGyro(100)
    
    #Define pin that will be used as output for the SPST relay switch
    channel = 21
    
    #GPIO.setwarnings(False) #write this line if we have tested that the pins are correct
    
    #GPIO numbers instead of board numbers
    GPIO.setmode(GPIO.BCM)
    
    #Setup GPIO pin for output mode
    GPIO.setup(channel, GPIO.OUT)
    print("GPIO set low")
    relay_off(channel)
    
    
    #Define parameters such as the acceleration threshold, how long the SPST relay should be enabled, and when the relay should open
    threshold = 0.5  # 5g should be enough
    open_count = 10  #should probably be around 1000
    count = 0
    wait = 10  # should probably be around 15 minutes

    # Initialize list for data in format: Time ; Accel_x [g] ;  Accel_y [g] ;  Accel_z [y] ; Roll [deg] ;  Pitch [deg] ;  Yaw [deg]   
    data = []
    
    # Run for TBD (should be around 15 minutes) secounds
    startTime = time.time()
    while(time.time() < (startTime + 10)):
        
        # Append data to the list
        t, ax, ay, az, r, p ,y = mpu.compFilter()
        print(" Time: " + str(t) \
            + " Roll: " + str(r) \
            + " Pitch: " + str(p) \
            + " Yaw: " + str(y) \
            + " Accel_x: " + str(ax) \
            + " Accel_y: " + str(ay) \
            + " Accel_z: " + str(az) )
        
        # List of values for this particular measurement
        values = [t, ax, ay, az, r, p ,y]
        
        # append to final list
        data.append(values)
        
        # Time before next measurement is taken
        time.sleep(0.4)
        
        
        # SPST Relay Code
        print(count)
        
        if relay_state == GPIO.LOW:
                #If acceleration exceeds threshold, start counting
                if az >= threshold:
                        count = count + 1

                        #If count is above a certain value open_count, switch on SPST relay via the GPIO.output, then wait for a duration defined by wait before switching off the relay
                        if count >= open_count:
                                relay_on(channel)
                                print("is open")
                                relay_state = GPIO.HIGH
                                print("Released")
                                
                                
                #If the acceleration was not long enough, set count to 0 and GPIO.output to low     
                else:
                        count = 0 
                        print("Not yet Released")
                
        
        if relay_state == GPIO.HIGH:
                continue
                
                
    # End
    print("Closing measurement procedure")

    # Print list 
    print(data)
    
    # Header
    header = ["Time", "Ax", "Ay", "Az", "Roll", "Pitch", "Yaw"]
    
    # Open a new CSV file in write mode
    start_of_measurement = datetime.datetime.now().strftime("%m-%d_%H:%M:%S.%f")
    filename = start_of_measurement + ".csv"
    
    with open(filename, mode="w") as csv_file:  # acceleration in g units and r,p,y in degrees
        # Create a CSV writer object
        csv_writer = csv.writer(csv_file)
        
        # Write each row of data to the CSV file
        csv_writer.writerow(header)
        csv_writer.writerows(data)
            
    # Print confirmation message
    print("Data saved to CSV file... Check!")
    
    # Set GPIO to LOW
    print("Turning of WD")
    relay_off(channel)
  
    # Ensure that pins are not left in an undefined state
    GPIO.cleanup()

# Main loop
if __name__ == '__main__':
	main()
