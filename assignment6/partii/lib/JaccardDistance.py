#####################################################
###             JaccardDistance.py                ###
#####################################################

import numpy as np

def jaccard_distance(str1, str2):
    '''
    jaccard distance measure
    :param str1: first list of words
    :param str2: second list of words
    :return: number
    '''

    inter = len(set(str1) & set(str2))
    uni = len(set(str1) | set(str2))

    return 1 - inter / uni

def average_jaccard_distance(str_x, str_list):
    '''
    average jaccard distance
    :param str_x: list of words
    :param str_list: list of list of words
    :return: float
    '''

    dist = list()
    for str_y in str_list:
        dist.append(jaccard_distance(str_x, str_y))

    return np.mean(dist)

def squareSum_jaccard_distance(str_x, str_list):
    '''
    square jaccard distance
    :param str_x: list of words
    :param str_list: list of list of words
    :return: float
    '''

    dist = list()
    for str_y in str_list:
        dist.append(jaccard_distance(str_x, str_y) ** 2)

    return sum(dist)

if __name__ == '__main__':
    print('JaccardDistance.py')
    # str1 = ['New', 'Aerial', 'Images', 'Show', 'Boston', 'Bombing', 'Suspect', 'In', 'Boat']
    # str2 = ['New', 'Aerial', 'Images', 'Show', 'Boston', 'Bombing', 'Suspect', 'In', 'Boat', 'Massachusetts', 'State', 'Police', 'released', 'five']
    # print(jaccard_distance(str1, str2))