import numpy as np
import random
import matplotlib.pyplot as plt

# Author: Sergey PLATONOV, HEPIA 2020-2021

class Cluster():

    def __init__(self, mean):
        self.data = []
        self.mean = mean
        self.prev_mean = None
        self.done = False

class K_means():

    MIN_VAL = 0
    MAX_VAL = 10000
    NB_POINTS = 200
    DEFAULT_K = 5
    EPSILON = 0.7


    def __init__(self, dataset=None, K=DEFAULT_K, points=NB_POINTS, min_val=MIN_VAL, max_val=MAX_VAL):
        """ generates 2D vectors at random, if no dataset provided, between the minimal and maximal value. Then K centroids are put at random."""

        self.K = K
        self.nb_points = points
        self.min_val = min_val
        self.max_val = max_val

        self.done = False

        self.clusters = [Cluster(self.__generate_random_2D_vector()) for _ in range(self.K)]
        self.dataset = self.__generate_random_set(self.nb_points) if not dataset else dataset

    def __generate_random_set(self, nb_points):
        """ Generates random set of 2D vectors """
        return [self.__generate_random_2D_vector() for _ in range(nb_points)]

    def __generate_random_2D_vector(self):
        return np.random.randint(self.min_val, self.max_val, size=(1, 2), dtype=int)


    def train(self):
        while not self.done:

            # calculate the distance of each point with respect to each mean
            distance_matrix = []
            for point in self.dataset:
                distances = []
                for cluster in self.clusters:
                    distance = np.linalg.norm(point - cluster.mean[0])
                    distances.append(distance)
                distance_matrix.append(distances)
            

            # find the minimal distance for each point, classifying them into respective clusters
            for i, distance_vector in enumerate(distance_matrix):
                cluster_nb = distance_vector.index(np.min(distance_vector))
                self.clusters[cluster_nb].data.append(self.dataset[i][0])

            # calculate the mean of the clusters
            for cluster in self.clusters:
                cluster.prev_mean = cluster.mean
                cluster.mean = self.__mean(cluster.data)

                cluster.done = np.linalg.norm(cluster.prev_mean - cluster.mean) < self.EPSILON

            # check whether the clusters have stopped moving - signalling the end of the algorithm
            self.done = all([cluster.done for cluster in self.clusters])
            

    def __mean(self, collection):
        average_x = sum([coord[0] for coord in collection]) / len(collection)
        average_y = sum([coord[1] for coord in collection]) / len(collection)

        return np.array([average_x, average_y])
        
    def display(self):
        colors = ['r', 'g', 'b', 'y', 'c']

        for i in range(self.K):
            plt.scatter([data[0] for data in self.clusters[i].data], [data[1] for data in self.clusters[i].data], c=colors[i])

        plt.show()

    def variance(self):
        variance = 0.0
        for cluster in self.clusters:
            for point in cluster.data:
                variance += np.linalg.norm(point - cluster.mean)
            
        variance /= self.nb_points

        print(variance)



    def __str__(self):
        pass

if __name__ == "__main__":

    k_means = K_means()

    k_means.train()

    k_means.variance()

    k_means.display()



