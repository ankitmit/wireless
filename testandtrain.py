import sys
import random
import math
import numpy as np
import  time
import requests
from requests.auth import HTTPDigestAuth
import json
import statistics
from scipy import stats
import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

alpha = 2.5
noise_count = 10
min_lon = -73.15
max_lon = -73.10
min_lat = 40.85
max_lat = 40.95

class row:
    def __init__(self, lat, lon, rssi):
        self.lat = lat
        self.lon = lon
        self.rssi = rssi

    def processRowInstance(self):
        row_text = ""
        row_text += str(self.lat)+'`'+str(self.lon)+'`'
        i = 0
        temp = ""
        l_rssi = len(self.rssi)
        while i < l_rssi:
            if(i == l_rssi -1):
                temp += str(self.rssi[i])
            else:
                temp += str(self.rssi[i]) + ','
            i += 1
        row_text += temp
        return row_text



def generateRandomTowers(dim, trans_count):
    range_num = int(math.pow(dim, 2))
    my_randoms = random.sample(xrange(range_num), 10)
    i = 0
    towers = []
    while i < trans_count:
        row = my_randoms[i] / dim
        col = my_randoms[i] % dim
        towers.append((row, col))
        i += 1
    return towers

def calculateDistance(i, j, k ,l):
    return math.sqrt((math.pow((k -i),2)) + math.pow((l-j), 2))


def generateDataForOneIteration(dim, trans_count, noise_std):
    towers = generateRandomTowers(dim, trans_count)
    grid = [[0 for x in range(dim)] for y in range(dim)]
    towers_cord = [[0 for x in range(dim)] for y in range(dim)]

    hor_cell_dist = (max_lat - min_lat) / dim
    vert_cell_dist = abs(min_lon - max_lon) / dim

    #Populate the co-ordinates where towers are present so that u dont calculate path loss for those cells
    i = 0
    while i < trans_count:
        towers_cord[towers[i][0]][towers[i][1]] = 1
        i += 1


    powers = []
    i = 0
    while i < dim:
        hor_loc = max_lat - (i * hor_cell_dist)
        j = 0
        while j < dim:
            vert_loc = min_lon + (j * vert_cell_dist)
            temp = []
            k = 0
            while k < trans_count:
                dist = calculateDistance(i, j, towers[k][0], towers[k][1])
                if dist != 0:
                    loss = 10 * alpha * math.log(dist)
                else:
                    loss = 0
                curr_power = 10 * math.log(0.016) - loss
                temp.append(curr_power)
                k += 1
            powers.append(row(hor_loc, vert_loc, temp))
            j += 1
        i += 1

    return powers

def splitData(consolidate_powers, dim):
    total = len(consolidate_powers)
    test_len = int(0.1 * total)
    test_data = []
    train_data = []
    test_loc = []
    train_loc = []

    test_ind = random.sample(xrange(total), test_len)
    i = 0
    while i < total:
        if test_ind.__contains__(i):
            test_data.append(consolidate_powers[i])
            cell_ind = i /10
            loc = str(cell_ind/dim) + "," + str(cell_ind % dim)
            test_loc.append(loc)
        else:
            train_data.append(consolidate_powers[i])
            cell_ind = i / 10
            loc = str(cell_ind / dim) + "," + str(cell_ind % dim)
            train_loc.append(loc)
        i += 1


    return test_data, train_data, test_loc, train_loc

def oneIteration(trans_count, resolution, std_deviation):
    dim = 200/resolution
    powers, towers_cord, towers = generateDataForOneIteration(dim, trans_count, std_deviation)
    testing_data, training_data, testing_data_loc, training_data_loc = splitData(powers, dim)
    model = GaussianNB()
    model.fit(training_data, training_data_loc)
    predicted = model.predict(testing_data)

    euclid_dist = []
    i = 0
    while i < len(predicted):
        x = int(predicted[i].split(',')[0])
        y = int(predicted[i].split(',')[1])
        t_x = int(testing_data_loc[i].split(',')[0])
        t_y = int(testing_data_loc[i].split(',')[1])
        dist = calculateDistance(t_x, t_y, x, y)
        euclid_dist.append(dist)
        i += 1

    return euclid_dist


def mainFunction():
    trans_count = [3,3,8,8]
    k = [5,5,5,5]
    std = [2,10,2,10]

    i = 0
    while i < 4:
        euclid_dist = oneIteration(trans_count[i], k[i], std[i])
        plt.plot(np.sort(euclid_dist), np.linspace(0, 1, len(euclid_dist), endpoint=False))
        i += 1
    plt.show()

def median(lst):
    return np.median(np.array(lst))

def secondQuestion():
    res = [1, 5, 10,15,20]
    #res = [25]
    std = [1, 2, 3, 4, 5, 10]

    result = []
    r = 0
    while r < len(res):
        i = 1
        result = [[0 for x in range(6)] for y in range(10)]
        while i < 10:
            j = 0
            while j < len(std):
                euclid_dist = oneIteration(i, res[r], std[j])
                med = median(euclid_dist)
                result[i-1][j] = med
                j += 1
            i += 1
        #print result
        plt.imshow(result, cmap='hot', interpolation="nearest")
        plt.colorbar()
        plt.show()
        r += 1
    #print len(result)


def thirdQuestion():
    result = [[0 for x in range(6)] for y in range(10)]
    res = [1,5,10,15,20]
    std = [1, 2, 3, 4, 5, 10]
    i = 0
    while i < 10:
        j  = 0
        while j < len(std):
            result[i][j] =res[random.randint(0,4)]
            j += 1
        i += 1
    plt.imshow(result, cmap='hot', interpolation="nearest")
    plt.colorbar()
    plt.show()

def generateData():
    resolution = 20
    dim = 200 / resolution
    powers = generateDataForOneIteration(dim, 5, 2.0)
    return powers

def deleteAllRowsFromTable():
    url = 'http://localhost:8080/wireless/rest/test/processRequest?request=1'
    myResponse = requests.get(url)
    if(myResponse.ok):
        print "All rows from table deleted\n"


def populateDataBase():
    deleteAllRowsFromTable()
    url = 'http://localhost:8080/wireless/rest/test/processRequest?request=2|'
    powers = generateData()
    total_len = len(powers)
    my_randoms = random.sample(xrange(total_len), total_len)
    i = 0
    j = 0
    while i < total_len:
        wait_time = random.randint(0,2)
        time.sleep(wait_time)
        row_text = powers[my_randoms[j]].processRowInstance()
        temp = url+row_text
        myResponse = requests.get(temp)
        print temp
        if not myResponse.ok:
            print "Insert failed for the row values " + row_text + "\nExiting the program\n"
            exit(0)
        i += 1
        j += 1

def testAPI():
    url = 'http://localhost:8080/wireless/rest/test/getContacts'
    myResponse = requests.get(url)
    print myResponse

def readDataFromTables():
    url = 'http://localhost:8080/wireless/rest/test/getDataFromTables'
    myResponse = requests.get(url)
    loc = []
    rssi_data = []
    if myResponse.ok:
        json_obj_list = json.loads(myResponse.content)
        for json_obj in json_obj_list:
            rssi_list = []
            temp_loc = str(json_obj['lat']) + ',' + str(json_obj['lon'])
            loc.append(temp_loc)
            temp_rssi = json_obj['rssi_vector'].split(',')
            for rssi in temp_rssi:
                rssi_list.append(float(rssi))
            rssi_data.append(rssi_list)

    return loc, rssi_data

def trainModel():
    testing_data_string = '-88.9349466896,-83.4428634423,-96.1269985008,-78.7983189868,-58.6803450814'
    test_data = []
    testing_data_list = testing_data_string.split(',')
    for data in testing_data_list:
        test_data.append(float(data))
    loc_training_data, rssi_training_data = readDataFromTables()
    model = GaussianNB()
    model.fit(rssi_training_data, loc_training_data)
    predicted = model.predict(test_data)
    print predicted

def getRowCount():
    url = 'http://localhost:8080/wireless/rest/test/getTableRowCount'
    myResponse = requests.get(url)
    if myResponse.ok:
        print myResponse.content

getRowCount()