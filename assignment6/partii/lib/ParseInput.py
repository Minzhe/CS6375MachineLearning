##############################################
###             readJson.py                ###
##############################################

import json
import re

def cleanString(string):
    '''
    clean string
    :param string: input string
    :return: list of words
    '''
    words = re.findall(r'[\w]+', string)
    words = [word.lower() for word in words if len(word) >= 2]

    return words

def readJson(jsonFile):
    '''
    Read json file
    :param jsonFile: json file
    :return: list of dict
    '''

    data = list()
    for line in open(jsonFile, 'r'):
        data.append(json.loads(line))

    tweets = dict()
    for item in data:
        tweets[item['id']] = cleanString(item['text'])

    return tweets

def readCenters(centerFile):
    '''
    Read center files
    :param centerFile: center file
    :return: list
    '''

    centers = list()
    for line in open(centerFile, 'r'):
        centers.append(int(line.strip('\n').strip(',')))

    return centers

if __name__ == '__main__':
    print('ParseInput.py')