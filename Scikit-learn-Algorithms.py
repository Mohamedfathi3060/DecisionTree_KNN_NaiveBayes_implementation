from main import xTrain,yTrain,xTest,yTest
from Custom_KNN import KNN as KNN_Scratch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib
from sklearn import tree
import pandas as pd
matplotlib.use("TkAgg")


# KNN
KNN = KNeighborsClassifier(n_neighbors=7)
KNN_ME = KNN_Scratch(k=7)

KNN.fit(xTrain, yTrain)
KNN_ME.fit(xTrain.to_numpy(), yTrain.to_numpy())

KNN_prediction = KNN.predict(xTest)
KNN_prediction_Scratch = KNN_ME.predict(xTest.to_numpy())


# Naive Bayes
NaiveBayes = GaussianNB()
NaiveBayes.fit(xTrain, yTrain)
NaiveBayes_predictions = NaiveBayes.predict(xTest)
print("Naive Bayes Score {Test Data}: ", NaiveBayes.score(xTest, yTest))


# Decision Tree
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree = DecisionTree.fit(xTrain, yTrain)
DecisionTree_predictions = DecisionTree.predict(xTest)
print("Decision Tree Score {Test Data}: ", DecisionTree.score(xTest,yTest))

# # Show Decision Tree plot
# fig = plt.figure(figsize=(12, 8))
# tree.plot_tree(DecisionTree, filled=True)
# plt.show()

# Metrics
metrics = {
    "Model": ["kNN", "KNN_Scratch", "Naive Bayes", "Decision Tree"],
    "Accuracy": [
        accuracy_score(yTest, KNN_prediction),
        accuracy_score(yTest, KNN_prediction_Scratch),
        accuracy_score(yTest, NaiveBayes_predictions),
        accuracy_score(yTest, DecisionTree_predictions),
    ],
    "Precision": [
        precision_score(yTest, KNN_prediction, average='macro'),
        precision_score(yTest, KNN_prediction_Scratch, average='macro'),
        precision_score(yTest, NaiveBayes_predictions, average='macro'),
        precision_score(yTest, DecisionTree_predictions, average='macro'),
    ],
    "Recall": [
        recall_score(yTest, KNN_prediction, average='macro'),
        recall_score(yTest, KNN_prediction_Scratch, average='macro'),
        recall_score(yTest, NaiveBayes_predictions, average='macro'),
        recall_score(yTest, DecisionTree_predictions, average='macro'),
    ],
}

results_df = pd.DataFrame(metrics)
print(results_df)
