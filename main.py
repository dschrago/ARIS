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
import subprocess


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
    
    
################################# Subprocess functions #################################

def execute_setup():
	process = subprocess.Popen("./setup.sh", executable="/home/bernoulli/simon")
	process.wait()
	
def execute_conf(filename, ip):
	process = subprocess.Popen(f"./smWDB -c /home/bernoulli/Configuration/{filename} l {ip}", shell=True)				
	process.wait()
	
def execute_h(filename, ip):
	process = subprocess.Popen(f"./smWDB -c /home/bernoulli/Configuration/{filename} h {ip}", shell=True)				
	process.wait()

def execute_k(filename, ip):
	process = subprocess.Popen(f"./smWDB -c /home/bernoulli/Configuration/{filename} k {ip}", shell=True)				
	process.wait()

def execute_data(filename, ip, newfilename, runtime_wd):
	process = subprocess.Popen(f"./smWDB -c /home/bernoulli/Configuration/{filename} -t {runtime_wd} -w {newfilename} w {ip}", shell=True)				
		

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
open_count1 = 10                     # when relay for acceleration should be closed, should probably be around 500

alt_threshold = 750                   # threshold for altitude, should be around 750 m
open_count2 = 500                     # when relay for altitude should be closed, should probably be around 500 

bmp.sea_level_pressure = 1013.25      # set the pressure at sea level for your location to get most accurate measurement
runningTime = 30                   # running time of experiment in seconds (should be around 15 minutes) 			  
temp_max = 70			   # if this value has been reached, turn of WD to prevent damage, should probably be around 70-75 

config_file = "test_2.json"       # filename of configuration
ip_wd = "192.168.5.150"		 # IP address of WD
newfilename = "Test"             # adapt filename of WD measurements
wd_time = 14			 # boot time of WD, should be around 11 s

wait_HV = 10			# wait this time for turning off HV of WD

rate = 0.05			      # rate of measurements of the sensors

    
################################# Data gathering #############################################

count1 = 0                             # counter for acceleration 
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
	temp = bmp.temperature
        
	# Adapt axis to our orientation of the BNO
	ax = -a_x
	ay =a_z 
	az = a_y
        
	# Adapt roll, yaw, pitch to our orientation of the BNO. Note that the BNO updates the axis directions by itself
	r = (roll-270)%360            # adapted roll to our setup for used convention 
	p = pitch                     # adapted pitch to our setup for used convention 
	y = (360-yaw)%360          # adapted yaw to our setup for used convention
	
	# Print BNO + BMP Data
	print(" Time: " + str(timestamp) \
            + " Roll: " + str(r) \
            + " Pitch: " + str(p) \
            + " Yaw: " + str(y) \
            + " Accel_x (m/s^2): " + str(ax) \
            + " Accel_y (m/s^2): " + str(ay) \
            + " Accel_z (m/s^2): " + str(az) \
	    + " Pressure (Pa): " + str(pressure) \
            + " Altitude (m): " + str(alt) \
	    + "Temperature (C)" + str(temp))
	    
            
	# SPST Relay Code
	
	print("Acceleration counter:", count1)
	print("Altitude counter:", count2)
	
        # If relay has not been released yet
	if relay_state == GPIO.LOW:
		#If acceleration exceeds threshold, start counting
		if az >= acc_threshold:
			count1 = count1 + 1

			#If count is above a certain value open_count, switch on SPST relay via the GPIO.output
			if count1 >= open_count1:
				
				#Activate relay and set relay_state to GPIO.HIGH
				relay_on(channel)
				relay_state = GPIO.HIGH   
				
				# Boot WD, requires wd_time 
				print("WD booting")
				time.sleep(wd_time)
				print("Boot time over")
			
				# Start setup of WD
				print("Start setup")
				
				print("End setup")
				
				# Start configuration of WD
				print("Start config")
				execute_conf(config_file, ip_wd)
				print("End config")
				
				# Turn on HV of WD
				print("Start turning on HV")
				execute_h(config_file, ip_wd)
				print("End turning on HV")
				
				# relay timer
				relay_timer = time.time() + runningTime 
				
				# WD starts gathering data
				print("Start of WD measurements")
				execute_data(config_file, ip_wd, newfilename, relay_timer)
								
		#If the acceleration was not long enough, set count to 0     
		else:
			count1 = 0 		
		
		#If altitude exceeds threshold, start counting
		if alt >= alt_threshold:
			count2 = count2 + 1

			#If count is above a certain value open_count2, switch on SPST relay via the GPIO.output
			if count2 >= open_count2:
				
				#Activate relay and set relay_state to GPIO.HIGH
				relay_on(channel)
				relay_state = GPIO.HIGH   
				
				# Boot WD, requires wd_time 
				print("WD booting")
				time.sleep(wd_time)
				print("Boot time over")
			
				# Start setup of WD
				print("Start setup")
				execute_setup()
				print("End setup")
				
				# Start configuration of WD
				print("Start config")
				execute_conf(config_file, ip_wd)
				print("End config")
				
				# Turn on HV of WD
				print("Start turning on HV")
				execute_h(config_file, ip_wd)
				print("End turning on HV")
				
				# relay timer
				relay_timer = time.time() + runningTime 
				
				# WD starts gathering data
				print("Start of WD measurements")
				execute_data(config_file, ip_wd, newfilename, relay_timer)      
                                
		#If the acceleration was not long enough, set count to 0     
		else:
			count2 = 0 			
			
	# If relay has been released	
	if relay_state == GPIO.HIGH:
		
		# Exit the loop after runningTime seconds have passed since relay_state was set to GPIO.HIGH
		if relay_timer is not None and time.time() >= relay_timer:
			# Turn off HV of WD
			print("Start turning off HV")
			execute_k(config_file, ip_wd)
			print("End turning off HV")
			
			
			break 
            
		# List of values for this particular measurement will be saved
		values = [timestamp, r, p ,y, alt, pressure,temp]
        
		# Append values to final list
		data.append(values)
	
	# Break out of loop if temperature overcomes temp_max to prevent damaging the WD
	if temp > temp_max:
		print("Start turning off HV")
		execute_k(config_file, ip_wd)
		print("End turning off HV")
		
		# List of values for this particular measurement will be saved
		values = [timestamp, r, p ,y, alt, pressure,temp]
        
		# Append values to final list
		data.append(values)
		
		break
    
        # Wait time rate until next measurement is taken
	time.sleep(rate)
	
	
################################# Saving Data to CSV #########################################
	
# Ending measurement gathering
print("Closing measurement procedure")


# Pop first row since not useful and print list 
data.pop(0)
print(data)
    
# Header: Time in seconds, Euler angles in degrees, Altitude in m and Pressure in Pa, Temperature in degrees Celsius 
header = ["Time (s)", "Roll (deg.)", "Pitch (deg.)", "Yaw (deg.)", "Altitude (m)", "Pressure (Pa)", "Temeprature (C)"]
    
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

time.sleep(wait_HV)

# Deactivate relay
print("Turning of WD")
relay_off(channel)
    
# Ensure that pins are not left in an undefined state
GPIO.cleanup()


