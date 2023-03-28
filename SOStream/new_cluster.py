from .utils import dist, weighted_mean
from math import exp

import numpy as np
import pandas as pd

class MicroCluster:

    def __init__(self, centroid, number_points = 1, radius = 0, label = None):
        self.number_points = number_points
        self.radius = radius
        self.centroid = centroid
        self.label = label 
    pass

def newCluster(vt, label=None):
    return MicroCluster(vt, label=label)


def merge_clusters(win_micro_cluster, overlaping_micro_clusters, merge_threshold):
    merged_cluster = None
    deleted_clusters = list()
    for micro_cluster in overlaping_micro_clusters:
        if dist(micro_cluster.centroid, win_micro_cluster.centroid) < merge_threshold:
            if len(deleted_clusters) == 0:
                deleted_clusters.append(win_micro_cluster)
                merged_cluster = MicroCluster(win_micro_cluster.centroid,
                                              number_points=win_micro_cluster.number_points,
                                              radius=win_micro_cluster.radius,
                                              label=win_micro_cluster.label)
            merged_cluster = merge(micro_cluster, merged_cluster)
            deleted_clusters.append(micro_cluster)
    return merged_cluster, deleted_clusters


def merge(cluster_a, cluster_b):
    new_cluster_centroid = weighted_mean(cluster_a.centroid, cluster_b.centroid, cluster_a.number_points, cluster_b.number_points)
    new_cluster_radius = dist(cluster_a.centroid, cluster_b.centroid) + max(cluster_a.radius, cluster_b.radius)
    new_cluster = MicroCluster(centroid=new_cluster_centroid,
                               number_points=cluster_a.number_points + cluster_b.number_points,
                               radius=new_cluster_radius,
                               label =cluster_a.label + cluster_b.label
                               )
    return new_cluster


def updateCluster(win_micro_cluster, vt, alpha, winner_neighborhood, label=None):
    win_micro_cluster.centroid = (win_micro_cluster.number_points * win_micro_cluster.centroid + vt) / (win_micro_cluster.number_points+1)
    win_micro_cluster.number_points += 1
    width_neighbor = win_micro_cluster.radius ** 2
    win_micro_cluster.label += label 
    for neighbor_micro_cluster in winner_neighborhood:
        influence = exp(-(dist(neighbor_micro_cluster.centroid, win_micro_cluster.centroid)/(2 * width_neighbor)))
        neighbor_micro_cluster.centroid = neighbor_micro_cluster.centroid + alpha*influence*(win_micro_cluster.centroid-neighbor_micro_cluster.centroid)
        neighbor_micro_cluster.label = neighbor_micro_cluster.label + alpha*influence*(win_micro_cluster.label + neighbor_micro_cluster.label) / 2
        # print(neighbor_micro_cluster.label, neighbor_micro_cluster.label, win_micro_cluster.label + neighbor_micro_cluster.label )
    # print()

def find_neighbors(win_microcluster, min_pts, model_t):
  if len(model_t) >= min_pts:
    win_dist = []
    for microcluster in model_t:
      win_dist.append(dist(microcluster.centroid, win_microcluster.centroid))
    win_dist.sort()
    idx_microclusters = np.argsort(win_dist)
    
    k_dist = win_dist[min_pts-1]
    win_microcluster.radius = k_dist
    win_nn = [model_t[idx] for idx in idx_microclusters[0:(min_pts)]]
    return win_nn
  else:
    return []

