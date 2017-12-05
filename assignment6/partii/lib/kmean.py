##############################################
###               kmean.py                 ###
##############################################

from lib.JaccardDistance import jaccard_distance, average_jaccard_distance, squareSum_jaccard_distance

def assignGroup(centers, points):
    '''
    Assign points to clusters
    :param centers: centers of clusters
    :param points: points to be clusters
    :return: dict
    '''

    ### check if centers are in points list
    if not set(centers) <= set(points.keys()):
        raise ValueError('Centers selected should be contained in the original data points!')

    ### find the nearest centers for each point
    assign = dict()
    for point in points.keys():
        dists = {center:jaccard_distance(points[point], points[center]) for center in centers}
        assign[point] = min(dists, key=dists.get)

    clusters = dict.fromkeys(centers)
    for center in centers:
        clusters[center] = [point for point, centroid in assign.items() if centroid == center]

    return clusters

def findCenters(clusters, points):
    '''
    Find the new centers with the clustered points
    :param clusters: clustered points
    :param points: points
    :return: list
    '''

    new_centers = list()
    for center in clusters.keys():
        ### subset points to current group
        grouped_point_words = {key: value for key, value in points.items() if key in clusters[center]}
        dist = {point: average_jaccard_distance(grouped_point_words[point], grouped_point_words.values()) for point in grouped_point_words.keys()}
        new_centers.append(min(dist, key=dist.get))

    return new_centers

def calculate_SSE(clusters, points):
    '''
    Calculate the SSE of clusters
    :param clusters: clustered points
    :param points: points
    :return: float
    '''

    dist2 = list()
    for center in clusters.keys():
        grouped_point_words = {key: value for key, value in points.items() if key in clusters[center]}
        dist2.append(squareSum_jaccard_distance(grouped_point_words[center], grouped_point_words.values()))

    return sum(dist2)