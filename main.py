import pandas as pandamodule
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import copy

class ModelData:
    labelEnc = LabelEncoder()
    scaler = StandardScaler()
    def __init__ (self):
        pass
    
    def loadCSV(self): 
        return pandamodule.read_csv('weather_forecast_data.csv')

    def checkMissingValues(self,data):
        return data.isnull().sum()
    
    def dropMissingValues(self,data):
        vData = copy.deepcopy(data)
        return vData.dropna()

    # with average
    def replaceMissingValues(self,data):
        vData = copy.deepcopy(data)
        return vData.fillna(vData.mean(numeric_only=True))

    def checkScale(self,data):
        return data.describe()

    def convertNumerical(self,data):
        numericalData = copy.deepcopy(data)
        y = copy.deepcopy(data["Rain"])
        self.labelEnc.fit(y)
        numericalData["Rain"] = self.labelEnc.transform(y)
        return numericalData

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
# print(missingValues)

droppedMissingData = DA.dropMissingValues(data)
replacedMissingData = DA.replaceMissingValues(data)
# print("Check Missing After Drop =>\n", DA.checkMissingValues((droppedMissingData)))
# print("Check Missing After Replace =>\n",DA.checkMissingValues((replacedMissingData)))

# print("Dropped Missing =>\n",droppedMissingData);
# print("Replaced with Average =>\n",replacedMissingData);

# # check whether numeric features have the same scale
dataScaleChecking = DA.checkScale(droppedMissingData)
_dataScaleChecking = DA.checkScale(replacedMissingData)
# print("Check Scalling After Drop =>\n",dataScaleChecking)
# print("Check Scalling After Replace =>\n",_dataScaleChecking)

# covert output name to numeric
numericData = DA.convertNumerical(droppedMissingData)
_numericData = DA.convertNumerical(replacedMissingData)
# print("numericalData After Drop=>\n",numericData)
# print("_numericalData After Replace=>\n",_numericData)

# the features and targets are separated
X,Y= DA.seperateTargets(numericData,"Rain")
_X,_Y= DA.seperateTargets(_numericData,"Rain")
# print("X =>\n", X)
# print("Y =>\n", Y)
# print("_X =>\n", _X)
# print("_Y =>\n", _Y)

# the data is shuffled and split into training and testing sets
xTrain,xTest,yTrain,yTest = DA.split(X,Y,0.2)
_xTrain,_xTest,_yTrain,_yTest = DA.split(_X,_Y,0.2)
# print("xTrain =>\n", xTrain)
# print("xTest =>\n", xTest)
# print("yTrain =>\n", yTrain)
# print("yTest =>\n", yTest)
# print("_xTrain =>\n", _xTrain)
# print("_xTest =>\n", _xTest)
# print("_yTrain =>\n", _yTrain)
# print("_yTest =>\n", _yTest)


# numeric features are scaled
xTrain = DA.scale(xTrain)
xTest = DA.transform(xTest)
_xTrain = DA.scale(_xTrain)
_xTest = DA.transform(_xTest)
# print("xTrain =>\n", xTrain)
# print("xTest =>\n", xTest)
# print("_xTrain =>\n", _xTrain)
# print("_xTest =>\n", _xTest)