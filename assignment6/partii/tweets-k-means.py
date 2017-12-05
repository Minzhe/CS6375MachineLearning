##################################################
###             tweets-k-means.py              ###
##################################################
# python tweets-k-means.py -s InitialSeeds.txt -i Tweets.json -o tweets-k-means-output.txt


import argparse
import random
from lib import ParseInput
from lib import kmean
from pprint import pprint

parser = argparse.ArgumentParser(description='tweet-k-menas')
parser.add_argument('-n', '--numberOfClusters', type=int, default=25, help='number of cluster')
parser.add_argument('-s', '--initialSeedsFile', help='initial seeds file')
parser.add_argument('-i', '--TweetsDataFile', help='input tweets data file')
parser.add_argument('-o', '--outputFile', help='output file')

args = parser.parse_args()
num_clusters = args.numberOfClusters
if num_clusters > 25:
    raise ValueError('number of cluster should be smaller than 25')
seeds = args.initialSeedsFile
tweets_path = args.TweetsDataFile
output_path = args.outputFile

print(num_clusters, seeds, tweets_path, output_path, sep='\n')

########## read json file ###########
data = ParseInput.readJson(jsonFile=tweets_path)
centers = ParseInput.readCenters(centerFile=seeds)
centers = random.sample(centers, num_clusters)

### first round
clusters = kmean.assignGroup(centers=centers, points=data)
new_centers = kmean.findCenters(clusters=clusters, points=data)

while set(new_centers) != set(centers):
    centers = new_centers.copy()
    clusters = kmean.assignGroup(centers=new_centers, points=data)
    new_centers = kmean.findCenters(clusters=clusters, points=data)
clusters = kmean.assignGroup(centers=new_centers, points=data)
# for center in new_centers:
#     for point in clusters[center]:
#         print(' '.join(data[point]))
#     print('--------------------')

sse = kmean.calculate_SSE(clusters=clusters, points=data)
with open(output_path, 'w') as f:
    for idx, item in enumerate(clusters.values()):
        text = str(idx+1) + ' ' + ','.join(map(str, item))
        print(text, file=f)
    print('-----------------', file=f)
    print('SSE: {}'.format(sse), file=f)