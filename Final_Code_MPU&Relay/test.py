import csv
import numpy as np

# CSV file with MPU data to be read out
filename = "MPU.csv"  # filename has to be adjusted

    # Open CSV file
with open(filename, mode="r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    next(csv_reader)
    data = []
    for row in csv_reader:
        data.append(row)