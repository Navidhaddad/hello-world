import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier

print("hello")

data = data = pd.read_csv('iris.csv', sep=';',header=None)
data = data.iloc[:,0:5]
data = data.dropna()
header = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'flowers']
data.columns = header

setosa = data[data.flowers=='Iris-setosa']
virginica = data[data.flowers=='Iris-virginica']
versicolor = data[data.flowers=='Iris-versicolor']

###Summary statistics
##Stats
# print(setosa.describe())
# print(virginica.describe())
# print(versicolor.describe())

##Skrewness and Kurtosis
# print(setosa.skew(), '\n', setosa.kurtosis())
# print(virginica.skew(), '\n', virginica.kurtosis())
# print(versicolor.skew(), '\n', versicolor.kurtosis())

###Vizualization
##Matrix for scatter
# sns.pairplot(data, hue="flowers", diag_kind="kde")
# scatter_matrix(data)
# plt.show()

##boxplot
# data.plot(kind='box')
# data.boxplot(by='flowers')

##Volinplot
# sns.violinplot(data=data, x='flowers', y='petal_length')

###Models
##Split data in traning and validation
test_index = np.random.choice(data.index, int(150/5), replace=False)
temp = data.loc[~data.index.isin(test_index)]
TDX = temp.iloc[:,0:4]
TDY = temp.iloc[:,4]
temp = data.loc[test_index]
valX =  temp.iloc[:,0:4]
ValY = temp.iloc[:,4]

##Make pridictions
np.random.seed(7)
knn = KNeighborsClassifier()
knn.fit(TDX,TDY)
pridictions = knn.predict(valX)
print(pridictions)
npValY = np.array(ValY)

n=0
for a, b in zip(npValY, pridictions):
	print("%s\n%s\n\n" % (a, b))
	if(a==b):
		n=n+1

print(n/len(npValY))
