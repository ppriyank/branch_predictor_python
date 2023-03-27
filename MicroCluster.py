import numpy as np
import sys
import numpy as np
from sklearn.utils import check_array
from copy import copy
from copy import deepcopy
from math import ceil
from sklearn.cluster import DBSCAN


class MicroCluster:
    def __init__(self, lambd, creation_time):
        self.lambd = lambd
        self.decay_factor = 2 ** (-lambd)
        self.mean = 0
        self.variance = 0
        self.sum_of_weights = 0
        self.creation_time = creation_time
        self.labels = {False:0 , True:0}

    def insert_sample(self, sample, y, weight):
        self.labels[y] += 1
        # print("--", self.sum_of_weights, self.labels)
        if self.sum_of_weights != 0:
            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight

            # Update mean
            old_mean = self.mean
            new_mean = old_mean + \
                (weight / new_sum_of_weights) * (sample - old_mean)

            # Update variance
            old_variance = self.variance
            new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) \
                + weight * (sample - new_mean) * (sample - old_mean)

            self.mean = new_mean
            self.variance = new_variance
            self.sum_of_weights = new_sum_of_weights
        else:
            self.mean = sample
            self.sum_of_weights = weight

    def radius(self):
        if self.sum_of_weights > 0:
            return np.linalg.norm(np.sqrt(self.variance / self.sum_of_weights))
        else:
            return float('nan')

    def center(self):
        return self.mean

    def weight(self):
        return self.sum_of_weights

    def __copy__(self):
        new_micro_cluster = MicroCluster(self.lambd, self.creation_time)
        new_micro_cluster.sum_of_weights = self.sum_of_weights
        new_micro_cluster.variance = self.variance
        new_micro_cluster.mean = self.mean
        new_micro_cluster.labels = self.labels
        return new_micro_cluster

class DenStream:

    def __init__(self, lambd=1, eps=1, beta=2, mu=2):
        """
        DenStream - Density-Based Clustering over an Evolving Data Stream with
        Noise.
        Parameters
        ----------
        lambd: float, optional
            The forgetting factor. The higher the value of lambda, the lower
            importance of the historical data compared to more recent data.
        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.
        Attributes
        ----------
        labels_ : array, shape = [n_samples]
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.
        Notes
        -----
        References
        ----------
        Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-Based
        Clustering over an Evolving Data Stream with Noise.
        """
        self.lambd = lambd
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.t = 0
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Online learning.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data
        y : Ignored
        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.
        Returns
        -------
        self : returns an instance of self.
        """
        X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))
        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, y, weight)
        return self

    def predict(self, X, sample_weight=None):
        
        X = check_array(X, dtype=np.float64, order="C")
        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        if self.p_micro_clusters == [] and self.o_micro_clusters == []:
            return None

        p_micro_cluster_centers = np.array([p_micro_cluster.center() for p_micro_cluster in self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in self.p_micro_clusters]

        o_micro_cluster_centers = np.array([o_micro_cluster.center() for o_micro_cluster in self.o_micro_clusters])
        o_micro_cluster_weights = [o_micro_cluster.weight() for o_micro_cluster in self.o_micro_clusters]

        centers = []
        if len(p_micro_cluster_centers) != 0  and len(o_micro_cluster_centers) ==0 :
            centers = p_micro_cluster_centers
            weights = p_micro_cluster_weights
            points = self.p_micro_clusters
        elif len(o_micro_cluster_centers) !=0 and len(p_micro_cluster_centers) ==0:
            centers = o_micro_cluster_centers
            weights = o_micro_cluster_weights
            points = self.o_micro_clusters
        else:
            centers = p_micro_cluster_centers
            centers = np.concatenate([p_micro_cluster_centers, o_micro_cluster_centers], 0)
            points = [x for x in self.p_micro_clusters] + [x for x in self.o_micro_clusters]


        # dbscan = DBSCAN(eps=5, algorithm='brute')
        # dbscan.fit(centers, sample_weight=weights)

        index, _ = self._get_nearest_micro_cluster(X, points)
        labels = points[index].labels
        y = labels[True] >= labels[False] 
        return y

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Lorem ipsum dolor sit amet
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data
        y : Ignored
        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster labels
        """
        X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))

        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
        
        p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                            p_micro_cluster in
                                            self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                   self.p_micro_clusters]
        dbscan = DBSCAN(eps=5, algorithm='brute')
        dbscan.fit(p_micro_cluster_centers,
                   sample_weight=p_micro_cluster_weights)

        y = []
        for sample in X:
            index, _ = self._get_nearest_micro_cluster(sample,
                                                       self.p_micro_clusters)
            y.append(dbscan.labels_[index])

        return y

    def _get_nearest_micro_cluster(self, sample, micro_clusters):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            current_distance = np.linalg.norm(micro_cluster.center() - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, y, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = deepcopy(micro_cluster)
            micro_cluster_copy.insert_sample(sample, y, weight)
            if micro_cluster_copy.radius() <= self.eps:
                micro_cluster.insert_sample(sample, y, weight)
                return True
        return False

    def _merging(self, sample, y, weight):
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, y, weight, nearest_p_micro_cluster)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, y, weight, nearest_o_micro_cluster)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, self.t)
                micro_cluster.insert_sample(sample, y, weight)
                self.o_micro_clusters.append(micro_cluster)

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def _partial_fit(self, sample, y, weight):
        self._merging(sample, y, weight)
        if self.t % self.tp == 0:
            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                     in self.p_micro_clusters if
                                     p_micro_cluster.weight() >= self.beta *
                                     self.mu]
            Xis = [((self._decay_function(self.t - o_micro_cluster.creation_time
                                          + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in
                   self.o_micro_clusters]
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]
        self.t += 1


    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")
        return sample_weight

class MicroCluster2:
    def __init__(self, lambd, creation_time):
        self.lambd = lambd
        self.decay_factor = 2 ** (-lambd)
        self.mean = 0
        self.variance = 0
        self.sum_of_weights = 0
        self.creation_time = creation_time
        self.labels = 0 

    def insert_sample(self, sample, y, weight):
        
        if self.sum_of_weights != 0:
            self.labels = self.labels * self.decay_factor + float(y)
            # Update sum of weights
            old_sum_of_weights = self.sum_of_weights
            new_sum_of_weights = old_sum_of_weights * self.decay_factor + weight

            # Update mean
            old_mean = self.mean
            new_mean = old_mean + \
                (weight / new_sum_of_weights) * (sample - old_mean)

            # Update variance
            old_variance = self.variance
            new_variance = old_variance * ((new_sum_of_weights - weight)
                                           / old_sum_of_weights) \
                + weight * (sample - new_mean) * (sample - old_mean)

            self.mean = new_mean
            self.variance = new_variance
            self.sum_of_weights = new_sum_of_weights
        else:
            self.mean = sample
            self.sum_of_weights = weight

    def radius(self):
        if self.sum_of_weights > 0:
            return np.linalg.norm(np.sqrt(self.variance / self.sum_of_weights))
        else:
            return float('nan')

    def center(self):
        return self.mean

    def weight(self):
        return self.sum_of_weights

    def __copy__(self):
        new_micro_cluster = MicroCluster2(self.lambd, self.creation_time)
        new_micro_cluster.sum_of_weights = self.sum_of_weights
        new_micro_cluster.variance = self.variance
        new_micro_cluster.mean = self.mean
        new_micro_cluster.label = self.label
        return new_micro_cluster

class DenStream2(DenStream):

    def predict(self, X, sample_weight=None):
        
        X = check_array(X, dtype=np.float64, order="C")
        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        if self.p_micro_clusters == [] and self.o_micro_clusters == []:
            return None

        p_micro_cluster_centers = np.array([p_micro_cluster.center() for p_micro_cluster in self.p_micro_clusters])
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in self.p_micro_clusters]

        o_micro_cluster_centers = np.array([o_micro_cluster.center() for o_micro_cluster in self.o_micro_clusters])
        o_micro_cluster_weights = [o_micro_cluster.weight() for o_micro_cluster in self.o_micro_clusters]

        centers = []
        if len(p_micro_cluster_centers) != 0  and len(o_micro_cluster_centers) ==0 :
            centers = p_micro_cluster_centers
            weights = p_micro_cluster_weights
            points = self.p_micro_clusters
        elif len(o_micro_cluster_centers) !=0 and len(p_micro_cluster_centers) ==0:
            centers = o_micro_cluster_centers
            weights = o_micro_cluster_weights
            points = self.o_micro_clusters
        else:
            centers = p_micro_cluster_centers
            centers = np.concatenate([p_micro_cluster_centers, o_micro_cluster_centers], 0)
            points = [x for x in self.p_micro_clusters] + [x for x in self.o_micro_clusters]


        # dbscan = DBSCAN(eps=5, algorithm='brute')
        # dbscan.fit(centers, sample_weight=weights)

        index, _ = self._get_nearest_micro_cluster(X, points)
        labels = points[index].labels
        y = labels >= 0
        return y

    def _merging(self, sample, y, weight):
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, y, weight, nearest_p_micro_cluster)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, y, weight, nearest_o_micro_cluster)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = MicroCluster2(self.lambd, self.t)
                micro_cluster.insert_sample(sample, y, weight)
                self.o_micro_clusters.append(micro_cluster)
