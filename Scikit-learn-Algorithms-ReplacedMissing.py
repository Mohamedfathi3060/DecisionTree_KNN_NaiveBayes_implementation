from main import _xTrain,_yTrain,_xTest,_yTest
from Custom_KNN import KNN as KNN_Scratch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score
import matplotlib.pyplot as plt
import matplotlib
from sklearn import tree
import pandas as pd
matplotlib.use("TkAgg")


# KNN
KNN = KNeighborsClassifier(n_neighbors=7)
KNN_ME = KNN_Scratch(k=7)

KNN.fit(_xTrain, _yTrain)
KNN_ME.fit(_xTrain.to_numpy(), _yTrain.to_numpy())

KNN_prediction = KNN.predict(_xTest)
KNN_prediction_Scratch = KNN_ME.predict(_xTest.to_numpy())


# Naive Bayes
NaiveBayes = GaussianNB()
NaiveBayes.fit(_xTrain, _yTrain)
NaiveBayes_predictions = NaiveBayes.predict(_xTest)
print("Naive Bayes Score {Test Data}: ", NaiveBayes.score(_xTest, _yTest))


# Decision Tree
DecisionTree = tree.DecisionTreeClassifier()
DecisionTree = DecisionTree.fit(_xTrain, _yTrain)
DecisionTree_predictions = DecisionTree.predict(_xTest)
print("Decision Tree Score {Test Data}: ", DecisionTree.score(_xTest,_yTest))

# Show Decision Tree plot
fig = plt.figure(figsize=(12, 8))
tree.plot_tree(DecisionTree, filled=True)
plt.show()

# Metrics
metrics = {
    "Model": ["kNN", "KNN_Scratch", "Naive Bayes", "Decision Tree"],
    "Accuracy": [
        accuracy_score(_yTest, KNN_prediction),
        accuracy_score(_yTest, KNN_prediction_Scratch),
        accuracy_score(_yTest, NaiveBayes_predictions),
        accuracy_score(_yTest, DecisionTree_predictions),
    ],
    "Precision": [
        precision_score(_yTest, KNN_prediction, average='macro'),
        precision_score(_yTest, KNN_prediction_Scratch, average='macro'),
        precision_score(_yTest, NaiveBayes_predictions, average='macro'),
        precision_score(_yTest, DecisionTree_predictions, average='macro'),
    ],
    "Recall": [
        recall_score(_yTest, KNN_prediction, average='macro'),
        recall_score(_yTest, KNN_prediction_Scratch, average='macro'),
        recall_score(_yTest, NaiveBayes_predictions, average='macro'),
        recall_score(_yTest, DecisionTree_predictions, average='macro'),
    ],
}

results_df = pd.DataFrame(metrics)
print(results_df)
