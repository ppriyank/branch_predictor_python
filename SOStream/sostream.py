from .utils import min_dist, dist, find_overlap
from .new_cluster import newCluster, merge_clusters, updateCluster, find_neighbors

class SOStream:

    def __init__(self, alpha = 0.1, min_pts = 10, merge_threshold = 27000):
        self.alpha = alpha
        self.min_pts = min_pts
        self.M = []
        self.merge_threshold = merge_threshold

    def process(self, vt, y ):
        winner_micro_cluster = min_dist(vt, self.M)
        new_M = self.M.copy()
        if len(new_M) >= self.min_pts:
            winner_neighborhood = find_neighbors(winner_micro_cluster, self.min_pts, new_M)
            if dist(vt, winner_micro_cluster.centroid) < winner_micro_cluster.radius:
                updateCluster(winner_micro_cluster, vt, self.alpha, winner_neighborhood, label=y * 1.0)
            else:
                new_M.append(newCluster(vt, label=y * 1.0 ))
            overlap = find_overlap(winner_micro_cluster, winner_neighborhood)
            if len(overlap) > 0:
                merged_cluster, deleted_clusters = merge_clusters(winner_micro_cluster, overlap, self.merge_threshold)
                for deleted_cluster in deleted_clusters:
                    new_M.remove(deleted_cluster)
                if merged_cluster is not None:
                    new_M.append(merged_cluster)
        else:
            new_M.append(newCluster(vt, label=y * 1.0 ))
        self.M = new_M
    pass

    def predict(self, vt):
        winner_micro_cluster = min_dist(vt, self.M)
        if winner_micro_cluster is None:
            return None
        return winner_micro_cluster.label > 0 
        