#This code is to read ROOT files and extract the relevant information as CSV/pandas to then later use for data analysis

import uproot
import pandas
import matplotlib.pyplot as plt
import awkward as ak
import numpy as np

#Open file 
file = uproot.open("TestRocketppl.root")

#Get the TTree file
tree = file["T"]

#List all the content
#print("Classnames: \n" + str(file.classnames()))

#To get the values of the hSampleSig34_0 channel
#print("Values of the channel: \n" + str(file["hSampleSig34_0"].values()))

#To convert to numpy
#print("Numpy array: \n" + str(file["hSampleSig34_0"].to_numpy()))

#Show different contents of TTree
#print(tree.keys())
#print(tree.values())
#print(tree.typenames())
#print(tree.show())

#Getting the content of the tree as pandas file
content = tree.arrays(["event", "timestamp", "amplitude"], library = "pd")
#print(content)

#Print content of TTree
print("Content of TTree: \n")
print(tree.keys())

#Get the branches as arrays
branches = tree.arrays()
print("Specific branch: \n")
print(branches["amplitude"])

#Getting specific value of a branch
print("Specific row of a branch: \n")
print(branches["TrigCell"][-1])

#Getting all information of all branches about a specific event
print("All information about a specific event: \n")
print(branches[0])

##################################################### All the branches as arrays #####################################################

tstmp = branches["timestamp"]                       #the timestamp w.r.t. 00:00 of the first day of DAQ
amp = branches["amplitude"]                         #maximum amplitude with respect to the estimated pedestal
time = branches["time"]                             #Constant fraction time
area = branches["area"]                             #charge integral in integration windows, not corrected for different bin widths (kept for legacy reasons)
area2 = branches["area2"]                           #charge integral in integration windows corrected for time bin width.
ped = branches["ped"]                               #pedestal level
Stdv = branches["Stdv"]                             #standard deviation (noise level) for the pedestal
tLE = branches["timeLE"]                            #Leading edge time
TrigCell = branches["TrigCell"]                     #Trigger Cell of the DRS4 chip, i.e. the cell where the chip was stopped.


#Convert all branches to a pd dataframe
amp_pd = ak.to_dataframe(amp)
tstmp_pd = ak.to_dataframe(tstmp)
time_pd = ak.to_dataframe(time)
area_pd = ak.to_dataframe(area)
ped_pd = ak.to_dataframe(ped)
Stdv_pd = ak.to_dataframe(Stdv)
tLE_pd = ak.to_dataframe(tLE)
area2_pd = ak.to_dataframe(area2)
TrigCell_pd = ak.to_dataframe(TrigCell)

#Print the pd of the different branches
print("Amplitudes: ")
print(amp_pd)

print("Timetamp: ")
print(tstmp_pd)

print("Time: ")
print(time_pd)

print("TimeLE: ")
print(tLE_pd)

#Accessing a specific column
print("Time of channel 34_0: ")
print(time_pd["34_0"])

print("Area: ")
print(area_pd["34_0"])

print("Area2: ")
print(area2_pd)

print("ped: ")
print(ped_pd)

print("Standard deviation: ")
print(Stdv_pd)

print("TrigCell: \n")
print(TrigCell_pd)

#Print data types of the branches
print("Datatypes of branches: ")
tree.show()


#Plot histogramm of a branch
#plt.hist(branches["timestamp"])
#plt.show()

#Get standard deviation of a branch (not for awkward arrays)
std_time = np.std(branches["timestamp"])
print(std_time)

#Getting length of the branch and number of events but NOT number of muons
length = len(branches)
#print("Number of events: " + str(length))

##################################################### Getting runtime of experiment #####################################################

t_min = ak.min(branches["timestamp"])
t_max = ak.max(branches["timestamp"])

t_exp = np.abs(t_min - t_max)/60
print("Runtime of the experiment in [min]: " + str(t_exp))

##################################################### Specific channel values #####################################################

t = tstmp_pd["values"]
amplitude = amp_pd["34_0"]
a = area_pd["34_0"]
a_2 = area2_pd["34_0"]
ped = ped_pd["34_0"]
trig = TrigCell_pd["34_0"]

##################################################### Getting Muon count ##########################################################

#Setting a certain threshold for the signal amplitude to filter muons to get a boolian array
a_2_th = a_2 > 0.4e-8
amplitude_th = amplitude > 0.4

#For all channels
a_2_th_all = area2_pd > 0.4e-8
amplitude_th_all = amp_pd > 0.4


#Checking each row individually if there is at least one channel with a signal to avoid overcounting
count = 0
for i in range(length):
    if(np.sum(amplitude_th_all.iloc[i]) >= 1):
        count = count + 1
    
    else: continue

print("Muon count of all channels: " + str(count))

#Getting the event vectors for the angle analysis as a boolean array
vec_0 = np.array(amplitude_th_all.iloc[24])
print(vec_0)

#Get index where we had a trigger
print("Indices where there was a trigger: ")
for i in range(length):
    vec_i = np.array(amplitude_th_all.iloc[i])
    if (np.sum(vec_i) == 1):
        print(i)
    else: continue


#Alternatively as matrix
vec = np.zeros((length, 18))
for i in range(length):
    vec[i,:] = np.array(amplitude_th_all.iloc[i])


#Muon count is then the sum of all the entries
muon_count = np.sum(amplitude_th)
print("Muon count: " + str(muon_count))

#Muon flux 
A = 2       #Area of detector in cm^2
flux = muon_count/(t_exp*A)
print("Muon flux per minute per cm^2 [min^-1 cm^-2]: " + str(flux))

plt.figure()
plt.xlabel("Time [s]")
#plt.ylabel("Amplitude")
plt.plot(t-42350, amplitude,label="Amplitude")
plt.plot(t-42350, a_2*10**8,label="Area")
plt.legend()

plt.figure()
plt.plot(t-42350, a_2)
plt.xlabel("Time [s]")
plt.ylabel("Area")

plt.figure()
plt.plot(t-42350, amplitude)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
#plt.show()

#amplitude_th_all.to_csv("Event trigger Matrix of all channels.csv", index=False)