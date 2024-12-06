import sys
from copy import deepcopy
import heapq
import math
import numpy


class KNN:
    def __init__(self,k):
        self.x_train = None
        self.y_train = None
        self.k = k

    def fit(self, x_train, y_train):
        self.x_train = deepcopy(x_train)
        self.y_train = deepcopy(y_train)
        if self.k > self.x_train.shape[0]:
            raise Exception("k is over than data points")

    def Euclidean_Distance(self, x_train, x_test):
        train_nFeatures = x_train.shape[0]
        test_nFeatures = x_test.shape[0]
        if train_nFeatures != test_nFeatures :
            raise IndexError("train features must be as test features")
        error = 0
        for i in range(train_nFeatures):
            error += (x_train[i] - x_test[i]) ** 2
        return math.sqrt(error)

    def predict(self, x_test):
        # y_test = numpy.zeros((x_test.shape[0], 1))
        y_test = ["#" for _ in range(x_test.shape[0])]
        for j in range(x_test.shape[0]):
            MINs = [(-sys.maxsize, 0)] * self.k
            for i in range(self.x_train.shape[0]):
                # -ve sign to achieve MAX-HEAP
                distance = self.Euclidean_Distance(self.x_train[i], x_test[j])
                heapq.heappushpop(MINs, (-distance, i))
            classA = 0
            classB = 0
            for err, idx in MINs:
                if self.y_train[idx] == 0:
                    classA += 1
                elif self.y_train[idx] == 1:
                    classB += 1
                else:
                    print("error")
            # print(MINs)
            # print("rain ", classB," Sample")
            # print("no-rain",classA, " Sample")
            if classA >= classB:
                y_test[j] = 0
            else:
                y_test[j] = 1
        return y_test


# ALG = KNN(xTrain.to_numpy(), yTrain.to_numpy(), k=1)
#
# predictions = ALG.predict(xTrain.to_numpy())
#
# y_ = yTrain.to_numpy()
#
# calc = 0
# for i in range(y_.shape[0]):
#     if predictions[i] == y_[i]:
#         calc += 1
# print(calc / y_.shape[0] )