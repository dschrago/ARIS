################################## Necessary packages ################################

import csv
import time
import board
import busio
import adafruit_bno055
import datetime
import numpy as np
import RPi.GPIO as GPIO
import os
import math
import adafruit_bmp3xx
import digitalio


################################## BMP & BNO initialisation ###########################

# I2C and BNO sensor initialisation
i2c = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_bno055.BNO055_I2C(i2c)

# SPI and BMP sensor initalisation
spi = board.SPI()
cs = digitalio.DigitalInOut(board.D5)
bmp = adafruit_bmp3xx.BMP3XX_SPI(spi,cs)


################################# Relay functions to turn on and off ###################

# Relay on
def relay_on(pin):
    print("Activating relay")
    GPIO.output(pin, GPIO.HIGH)     #set the relay pin to high (5V) to activate the relay

# Relay off
def relay_off(pin):
    print("Deactivating relay")
    GPIO.output(pin, GPIO.LOW)      #set the relay pin to low (0V) to deactivate the relay


################################# Main Code #############################################

# Initialize list for data in format: Time [s]; Roll [deg]; Pitch [deg]; Yaw [deg]; Altitude [m]; Pressure [Pa]
data = []

# Define pin that will be used as output for the SPST relay switch
channel = 17
    
# GPIO.setwarnings(False) #write this line if we have tested that the pins are correct
    
# GPIO numbers instead of board numbers
GPIO.setmode(GPIO.BCM)
    
# Setup GPIO pin for output mode
GPIO.setup(channel, GPIO.OUT)
    
# Set initial state of the relay to be deactivated
relay_off(channel)
    
# State of relay
relay_state = GPIO.LOW

    
################################ Parameters to be defined ##############################

# Define parameters such as the acceleration and altitude threshold, how long the SPST relay should be enabled, and when the relay should close
# Backup: relay closes if certain altitude has been reached

acc_threshold = 4                     # 40 m/s^2 should be enough
open_count = 1000                     # when relay for acceleration should be closed, should probably be around 1000

alt_threshold = 750                   # threshold for altitude, should be around 750 m
open_count2 = 500                     # when relay for altitude should be closed, should probably be around 500 

bmp.sea_level_pressure = 1013.25      # set the pressure at sea level for your location to get most accurate measurement
runningTime = 60*15                   # running time of experiment in seconds (should be around 15 minutes) 

rate = 0.01			      # rate of measurements of the sensors

# Note: check adapted yaw, might need to modify +45 to -45 depending on decision of cube's orientation
    
################################# Data gathering #############################################

count = 0                             # counter for acceleration 
count2 = 0			      # counter for altitude
relay_timer = None		      # purpose to determine when the additional runningTime period has elapsed after the relay is activated

while True:
        
	# Timestamp of measurement
	timestamp = time.time() 
        
	# Yaw, Pitch, Roll 
	(yaw,pitch,roll) = sensor.euler   
        
	# Acceleration in x,y,z axis
	(a_x,a_y,a_z) = sensor.acceleration
	
	# Pressure in Pa and Altitude in m
	pressure = bmp.pressure
	alt = bmp.altitude
        
	# Adapt axis to our orientation of the BNO
	ax = -a_x
	ay =a_z 
	az = a_y
        
	# Adapt roll, yaw, pitch to our orientation of the BNO. Note that the BNO updates the axis directions by itself
	r = (roll-270)%360            # adapted roll to our setup for used convention 
	p = pitch                     # adapted pitch to our setup for used convention 
	y = (360-yaw+45)%360          # adapted yaw to our setup for used convention. We also considered the 45 deg. offset with the cubic structure
	
	# Print BNO + BMP Data
	print(" Time: " + str(timestamp) \
            + " Roll: " + str(r) \
            + " Pitch: " + str(p) \
            + " Yaw: " + str(y) \
            + " Accel_x (m/s^2): " + str(ax) \
            + " Accel_y (m/s^2): " + str(ay) \
            + " Accel_z (m/s^2): " + str(az) \
	    + " Pressure (Pa): " + str(pressure) \
            + " Altitude (m): " + str(alt))
	    
            
	# SPST Relay Code
	
	#print("Acceleration counter:", count)
	#print("Altitude counter:", count2)
	
        # If relay has not been released yet
	if relay_state == GPIO.LOW:
		#If acceleration exceeds threshold, start counting
		if az >= acc_threshold:
			count = count + 1

			#If count is above a certain value open_count, switch on SPST relay via the GPIO.output
			if count >= open_count:
				relay_on(channel)
				relay_state = GPIO.HIGH   
				relay_timer = time.time() + runningTime  
                                
		#If the acceleration was not long enough, set count to 0     
		else:
			count = 0 
			#print("Not yet Released")
		
		
		#If altitude exceeds threshold, start counting
		if alt >= alt_threshold:
			count2 = count2 + 1

			#If count is above a certain value open_count2, switch on SPST relay via the GPIO.output
			if count2 >= open_count2:
				relay_on(channel)
				relay_state = GPIO.HIGH
				relay_timer = time.time() + runningTime       
                                
		#If the acceleration was not long enough, set count to 0     
		else:
			count2 = 0 
			#print("Not yet Released")
			
			
	# If relay has been released	
	if relay_state == GPIO.HIGH:
		
		# Exit the loop after runningTime seconds have passed since relay_state was set to GPIO.HIGH
		if relay_timer is not None and time.time() >= relay_timer:
			break 
            
		# List of values for this particular measurement will be saved
		values = [timestamp, r, p ,y, alt, pressure]
        
		# Append values to final list
		data.append(values)
    
        # Wait time rate until next measurement is taken
	time.sleep(rate)
	
	
################################# Saving Data to CSV #########################################
	
# Ending measurement gathering
print("Closing measurement procedure")

# Print list 
print(data)
    
# Header: Time in seconds, Euler angles in degrees, Altitude in m and Pressure in Pa
header = ["Time", "Roll", "Pitch", "Yaw", "Altitude", "Pressure"]
    
# Open a new CSV file in write mode
start_of_measurement = datetime.datetime.now().strftime("%m-%d_%H:%M:%S.%f")
filename = start_of_measurement + ".csv"
    
with open(filename, mode="w") as csv_file: 
	# Create a CSV writer object
	csv_writer = csv.writer(csv_file)
        
	# Write each row of data to the CSV file
	csv_writer.writerow(header)
	csv_writer.writerows(data)
            
# Print confirmation message
print("Data saved to CSV file... Check!")
    
    
################################# Turning off WD #############################################
    
# Deactivate relay
print("Turning of WD")
relay_off(channel)
    
# Ensure that pins are not left in an undefined state
GPIO.cleanup()


