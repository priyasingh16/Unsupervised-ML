from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from mnist import MNIST
import numpy as np

newsgroups_train = fetch_20newsgroups(subset='train')
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train.data)

print("Pairwise similarity/distance matrix - cosine or simple dot product:", cosine_similarity(vectors))

print("Pairwise similarity/distance matrix - Euclidian distance:", euclidean_distances(vectors))

mndata = MNIST('MNIST/')
mndata.gz = True
images_train , labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()
images = images_train + images_test

filename1 = 'euclidean_distance.npy'

euclidean = np.memmap(filename1, dtype='float32', mode='w+', shape=(70000, 70000))
for i in range(70):
    x = 1000*i
    y = 1000*(i+1)

    temp = euclidean_distances(images[x:y],images)
    euclidean[x:y] = temp
    # print temp

print("Pairwise similarity/distance matrix - Euclidian distance:", euclidean)
# f = open(filename, 'r')

# print euclidean

filename2 = 'cosine_similarity.npy'

cosine = np.memmap(filename2, dtype='float32', mode='w+', shape=(70000,70000))
for i in range(70):
    x = 1000*i
    y = 1000*(i+1)

    temp = cosine_similarity(images[x:y],images)
    cosine[x:y] = temp
    # print temp
print("Pairwise similarity/distance matrix - cosine or simple dot product:", cosine)
