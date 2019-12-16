# Anant Ahuja
# axa9357
#--------#---------#---------#---------#--------#--------#---------#---------#---------#--------#
import sys
import csv
import numpy as np
import random
from collections import Counter
max_iterations = 300
#--------#---------#---------#---------#--------#--------#---------#---------#---------#--------#
def main():

    # Path supplied by command line argument
    if len(sys.argv) != 3:
        print("Usage: python3 asg3.py iris.data k")
        return

   # Parsing the validating dataset and loading it
    dataset = open(sys.argv[1])
    dataset_reader = csv.reader(dataset)
    data = list(dataset_reader)
    dataset.close()


    # Converting the labels from strings to integer values 1,2 and 3
    for row in data:
        if row[4] == 'Iris-setosa':
            row[4] = 0
        elif row[4] == 'Iris-versicolor':
            row[4] = 1
        else:
            row[4] = 2

    seed = random.random()

    k = int(sys.argv[2])

    centroids, labels = kmeans(data, k)

    correct_count = 0
    total_count = len(data)

    setosa_list = labels[0:50]
    versicolor_list = labels[50:100]
    virginica_list = labels[100:150]

    setosa_label_freq = Counter(setosa_list).most_common()
    versicolor_label_freq = Counter(versicolor_list).most_common()
    virginica_label_freq = Counter(virginica_list).most_common()

    correct_count = setosa_label_freq[0][1] + versicolor_label_freq[0][1] + virginica_label_freq[0][1]

    accuracy = (correct_count / total_count) * 100

    print('Centroids = {}'.format(centroids) )
    print('Labels = {}'.format(labels))
    print('Accuracy = {}'.format(accuracy))

def kmeans(data, k):

    # Forgy method of initializing centroids
    #random.seed(seed)
    centroids = random.sample(data, k)
    centroids = np.array(centroids, dtype=np.float64)
    centroids = centroids[ : , 0 : 4]

    data = np.array(data, dtype=np.float64)
    data = data[ : , 0 : 4]

    num_features = len(data[0])

    iterations = 0
    old_centroids = None

    # Run main kmeans algorithm
    while not stop(old_centroids, centroids, iterations):

        old_centroids = centroids
        iterations += 1

        labels = get_labels(data, centroids)

        centroids = get_centroids(data, labels, k)

    labels = get_labels(data, centroids)

    return centroids, labels

# kmeans terminates either because it has run maximum number of iterations OR centroids stop changing
def stop(old_centroids, centroids, iterations):
    if (iterations > max_iterations):
        return true
    return np.array_equal(old_centroids, centroids)

def get_labels(data, centroids):

    labels = []

    for datapoint in data:

        distance_to_centroid = []

        for centroid in centroids:
            distance_to_centroid.append(np.sqrt(sum((centroid - datapoint) ** 2)))

        labels.append(np.argmin(distance_to_centroid))

    return labels

def get_centroids(data, labels, k):

    clusters = {}

    centroids = []

    for i in range(k):

        clusters[i]= []

        for datapoint, label in zip(data, labels):

            if label == i:
                clusters[i].append(datapoint)

        centroids.append(np.mean(clusters[i]))

    return centroids

if __name__ == '__main__':
    main()