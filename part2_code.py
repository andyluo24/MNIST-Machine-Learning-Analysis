import sys
import sklearn
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from keras.datasets import mnist
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train_array = np.asarray(x_train)
x_test_array = np.asarray(x_test)
y_train_array = np.asarray(y_train)
y_test_array = np.asarray(y_test)

## Use k-means to build a dictionary of 10 x 10 image patches clustering in 50 
## centers. (part 1)

def splitingPatch(array):
    patches = np.zeros((60000, 16,100))
    for row in range(60000):
      for cell in range(16):
        X = 0
        for n in range(10):
          for m in range(10):
            patches[row][cell][x] = array[row][28*n+6*cell+m]
            X += 1
    return patches

# convert the dimension of the original datasets to 1D
X = x_train_array
Y = y_train_array
X = X.reshape(60000,784)
print(X.shape)

# noramalize the data to 0-1
def chooseOne(array):
    temp = np.zeros((60000,100))
    idx = np.random.randint(16, size = 1)
    for row in range(60000):
      temp[row] = array[row][idxGrid,:]
    return temp

# 4x4 grids 16 pathces totoal
patches = splitingPatch(X)
idx = np.random.randint(60000, size = 6000)
idxGrid = np.random.randint(16, size = 1)

# convert patches 60000,16,100 to 60000,100; 1 among 16
newpatches = chooseOne(patches)
print(patches.shape)
predictPatches = patches.reshape(960000,100)
print(predictPatches.shape)

# now convert patches 60000,100 to 6000,100; 6000 among 60000
idx = np.random.randint(60000, size = 6000)
RandomInput = newpatches[idx,:]
print(RandomInput.shape)

# uniformaly select 6000 random subsets from 60000
idx = np.random.randint(60000, size = 6000)
RandomInput = newpatches[idx,:]
RandomInput = RandomInput.reshape(len(RandomInput),-1)

# Initialize KMeans Model
Kmean = KMeans(n_clusters = 50)
Kmean.fit(RandomInput)
Kmean.cluster_centers_
print(Kmean.labels_.shape)

# assign our new test_array (60000, )
sample_test = x_test_array
sample_test = sample_test.reshape(1,-1)
new_dict = defaultdict(list)

result = Kmean.predict(predictPatches)
print(predictPatches.shape)
print(result.shape)

for i in range(960000):
  new_dict[result[i]].append(predictPatches[i])

dictModels = defaultdict(tuple)
Kmean2 = KMeans(n_clusters = 50)

for i in range(50):
  dictModels[i] = Kmean2.fit(new_dict[i])

## Create a set of 10 x 10 patches on an overlapping 4 x 4 grid for each query 
## image and result in a total of 144 patches for each test image. Use the 
## dictionary to find the closest center to each patch and construct a histogram 
## of patches for each test image. 
## (part 2)

# Extend the image 28x28 to 30x30 then split it into 9 patches.
def Expanding144(array, idx):
    newExpand = np.zeros((idx, 900))
    for row in range(idx):
      for n in range(28):
        for m in range(28):
          newExpand[row][31+30*n+m] = array[row][28*n + m]
    return newExpand

# Spliting to 144 patches
def spliting144(array, idx):
    patches = np.zeros((idx,144,100))
    for row in range(idx):
      for cell in range(16):
        for i in range(3):
          for j in range(3):
            x = 0
            for n in range(10):
              for m in range(10):
                patches[row][(9*cell)+(3*i)+j][x] = array[row]
                x += 1
    return patches

# randomly choosing 6000 out of 60000
temp = Expanding144(X, 60000)
temp = spliting144(temp, 60000)
Input1 = temp

# predicting first level of the Kmean for training
histogram = []
for i in range(60000):
    if i%500 == 0:
      print(i)
    resultLevel2 = np.zeros(2500)
    result1 = Kmean.predict(Input1[i])
    for j in range(144):
      result2 = dictModels[result1[j]].predict(Input1[j])
      resultLevel2[result1[j]*50 + result2] += 1
    histogram.append(resultLevel2)

# prepare data for y_test
new_x_test = x_test.reshape(10000,784)
test_input = Expanding144(new_x_test, 10000)
test_input = spliting144 (test_input, 10000)

# predicting first level of the Kmean for testing 
histogram1 = []
for i in range (10000):
  if i % 500 == 0:
    print(i)
  resultLevel2 = np.zeros(2500)
  result1 = Kmean.predict(test_input[i])

  for j in range(144):
    result2 = dictModels[result1[j]].predict(test_input[j])
    resultLevel2[result1[j] * 50 + result2] += 1
  histogram1.append(resultLevel2)

## Train a decision forest classifier based on the historgram of patches and 
## check the accuracy of this classifier on the testing data. (part 3)

print(y_train_array.shape)
new_y_train = y_train_array
print(y_test.shape)

# train the decision tree classifier
model = RandomForestClassifier(n_estimators = 90, max_depth = 60)
model.fit(histogram, new_y_train)

# test the decision tree classifier and evaluate the accuracy
y_predict = model.predict(histogram1)
accuracy_score(y_predict, y_test)
