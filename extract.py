import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

'''
Data provided by UCI Machine Learning Repositoy on Thoracic Surgery. 
Data Source: https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data

Goals:
1. convert categorical data into numerical values 
2. convert T/F to 1/0

Author: Ly Sung
Date: March 20th 2019
'''

'''
convert categorical data into numerical values 
ie: DGN4 = 4, T = 1, OC11 = 11 
'''
def mapping(data):
    operator = {
        "DGN1": 1,
        "DGN8": 8,
        "DGN5": 5,
        "DGN6": 6,
        "DGN4": 4,
        "DGN2": 2,
        "DGN3": 3,
        "OC13": 13,
        "OC12": 12,
        "OC14": 14,
        "OC11": 11,
        "PRZ0": 0,
        "PRZ1": 1,
        "PRZ2": 2,
        "T": 1,
        "F": 0
    }

    for i in range(len(data)):
        if data[i] in operator:
            data[i] = operator[data[i]]
        else: 
            data[i] = float(data[i])

def extract_data():
    file_name = "ThoraricSurgery.arff.csv"
    data = []
    target = []
    csv_reader = csv.reader(open(file_name), delimiter=",")

    print("Extracting data...")
    for row in csv_reader:
        mapping(row)
        target.append(row[-1])
        row = list(preprocessing.scale(row[0:-1]))
        data.append(row)
        

    print("Finished extracting data!")

    return data, target
