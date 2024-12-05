import pandas as pandamodule
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

class ModelData:
    # featuresEnc = LabelEncoder()
    labelEnc = LabelEncoder()
    scaler = StandardScaler()
    def __init__ (self):
        pass
    
    def loadCSV(self): 
        return pandamodule.read_csv('weather_forecast_data.csv')

    def checkMissingValues(self,data):
        return data.isnull().sum()
    
    def dropMissingValues(self,data):
        return data.dropna()

    # with average
    def replaceMissingValues(self,data):
        return data.fillna(data.mean(numeric_only=True))

    def checkScale(self,data):
        return data.describe()
    
    # def showPairPlot(self,data):
    #     sns.pairplot(data, diag_kind='hist')
    #     plt.show()

    # def showHeatMap(self,data):
    #     numeric_data = data.select_dtypes(include=['number'])
    #     correlation_matrix = numeric_data.corr()
    #     plt.figure(figsize=(10, 8))
    #     sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    #     plt.title('Correlation Heatmap')
    #     plt.show()

    def seperateTargets(self,data,targets):
        X = data.drop(targets, axis=1)
        Y = data[targets]
        return X,Y
    
    
    def split(self, X, Y, testSize):
        xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size=testSize, random_state=42)
        return xTrain,xTest,yTrain,yTest,
    
    def scale(self,data):
        scaled_features = self.scaler.fit_transform(data)
        return pandamodule.DataFrame(scaled_features, columns=data.columns)

    def transform(self,data):
        scaled_features = self.scaler.transform(data)
        return pandamodule.DataFrame(scaled_features, columns=data.columns)

    

DA = ModelData()

# Load the "weather_forecast_data" dataset
data = DA.loadCSV()

# check whether there are missing values
missingValues = DA.checkMissingValues(data)
print(missingValues)

# droppedMissingData = DA.dropMissingValues(data)
replacedMissingData = DA.replaceMissingValues(data)
# print("check missing after drop =>\n", DA.checkMissingValues((droppedMissingData)))
print("check missing after replace =>\n",DA.checkMissingValues((replacedMissingData)))

# print("Dropped Missing =>\n",droppedMissingData);
print("Replaced with Average =>\n",replacedMissingData);

# # check whether numeric features have the same scale
dataScaleChecking = DA.checkScale(data)
print("Check Scalling =>\n",dataScaleChecking)

# # visualize a pairplot in which diagonal subplots are histograms
# # DA.showPairPlot(data)

# # visualize a correlation heatmap between numeric columns
# # DA.showHeatMap(data)

# the features and targets are separated
X,Y= DA.seperateTargets(replacedMissingData,"Rain")
print("X =>\n", X)
print("Y =>\n", Y)

# the data is shuffled and split into training and testing sets
xTrain,xTest,yTrain,yTest = DA.split(X,Y,0.3)
print("xTrain =>\n", xTrain)
print("xTest =>\n", xTest)
print("yTrain =>\n", yTrain)
print("yTest =>\n", yTest)


# numeric features are scaled
xTrain = DA.scale(xTrain)
xTest = DA.transform(xTest)
print("xTrain =>\n", xTrain)
print("xTest =>\n", xTest)