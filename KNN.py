from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from mnist import MNIST
from scipy.sparse import csr_matrix
import numpy as np
import collections



def knn(train_data, labels_train, test_data, k):
    sim = cosine_similarity(test_data, train_data)
    predictions = []
    for i in xrange(0, len(sim)):
        topK_labs = []
        arr = sorted(sim[i], reverse=True)
        for val in arr[:k]:
            topK_labs.append(labels_train[np.where(sim[i]==val)[0][0]])

        freq = collections.Counter(topK_labs)
        max = 0;
        lab = -1;
        for key, value in freq.iteritems():
            if value > max:
                max = value
                lab = key
        predictions.append(lab)
    return predictions

def accuracy(test_labels, test_prediction):
    count = 0;
    for x,y in zip(test_labels, test_prediction):
        if x == y:
            count = count + 1
    return (float(count) * 100.0) / float(len(test_prediction))



mndata = MNIST('MNIST/')
mndata.gz = True
images_train , labels_train = mndata.load_training()
images_test, labels_test = mndata.load_testing()
mnist_pred = knn(images_train, labels_train, images_test, 5)
print mnist_pred
print accuracy(labels_test, mnist_pred)


vectorizer = TfidfVectorizer()

newsgroups_train = fetch_20newsgroups(subset='train')
vectors_train = vectorizer.fit_transform(newsgroups_train.data)
label_train = newsgroups_train.target

newsgroups_test = fetch_20newsgroups(subset='test')
vectors_test = vectorizer.transform(newsgroups_test.data)
label_test = newsgroups_test.target


news = knn(vectors_train, label_train, vectors_test, 5)
print news
print accuracy(label_test, news)
