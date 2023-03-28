from math import sqrt
import numpy as np
import pandas as pd

def dist(v1, v2):
    return np.sqrt(((v1 - v2) ** 2).sum())


def weighted_mean(a, b, wieght_a, wieght_b):
    return ((wieght_a * a + wieght_b * b)/(wieght_a + wieght_b))


def min_dist(vt, micro_clusters):
    micro_cluster_min_dist = float('inf')
    min_micro_cluster = None
    for micro_cluster in micro_clusters:
        dist_to_micro_cluster = dist(vt, micro_cluster.centroid)
        if dist_to_micro_cluster <= micro_cluster_min_dist:
            micro_cluster_min_dist = dist_to_micro_cluster
            min_micro_cluster = micro_cluster
    
    return min_micro_cluster


def fadingAll(M, t_current, var_lambda, fadeThreshold):
  M_copy = M.copy()
  for microcluster in M_copy:
    t = t_current - microcluster['t']
    fading = microcluster.number_points*2**(-var_lambda*t)
    if fading < fadeThreshold:
      M.remove(microcluster)




def find_overlap(win, win_nn):
  overlap = []
  for microcluster in win_nn:
    if win is not microcluster:
      if dist(win.centroid, microcluster.centroid) - (win.radius + microcluster.radius) < 0 :
        overlap.append(microcluster)
  return overlap
