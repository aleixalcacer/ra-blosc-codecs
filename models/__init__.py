from archetypes import AA as AA_archetypes
from sklearn.cluster import KMeans as KMeans_cluster
from .cmeans import CMeans as CMeans_fuzzy
import numpy as np

class AA(AA_archetypes):
    @property
    def prototypes(self):
        return self.archetypes_


class KMeans(KMeans_cluster):
    @property
    def prototypes(self):
        return self.cluster_centers_

    def transform(self, X):
        kmeans_labels = self.predict(X)
        kmeans_weights = np.zeros((kmeans_labels.size, self.n_clusters))
        kmeans_weights[np.arange(kmeans_labels.size), kmeans_labels] = 1
        return kmeans_weights

class CMeans(CMeans_fuzzy):
    @property
    def prototypes(self):
        return self.centroids_

