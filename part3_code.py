import numpy as np
import matplotlib.pyplot as plt
import copy
from keras.datasets import mnist

# obtain the MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def retrieve_images(dataset, a = 500, b = 28, c = 28):
    imgs = np.zeros((a,b,c))
    for i in range(500):
      for j in range(len(dataset[i])):
        for k in range(len(dataset[i][j])):
          imgs[i][j][k] = dataset[i][j][k]/255
    return imgs

def binarize(dataset):
    for i in range(len(dataset)):
      for j in range(len(dataset[i])):
        for k in range(len(dataset[i][j])):
          dataset[i][j][k] = 1 if dataset[i][j][k] > 0.5 else -1
    return dataset

def noise(dataset1, dataset2):
    for i in range(len(dataset1)):
      for j in range(len(dataset1[i])):
          for k in range(len(dataset1[i][j])):
            dataset1[i][j][k] = -dataset2[i][j][k] if np.random.binomial(1,0.98) != 1 else dataset2[i][j][k]
    return dataset1

def twodarr(num):        ## create a 2d array of empty brackets
    arr = []
    for i in range(num):
      arr.append([])
    return arr

def denoise(dataset, θ_i, θ_j):
    denoise_img = twodarr(500)
    for a in range(len(dataset)):
      pixel = np.zeros((dataset.shape[1], dataset.shape[2])) + 0.5  
      pre_pixel = np.copy(pixel)
      for b in range(30):
        for c in range(len(dataset[a])):
          for d in range(len(dataset[a][c])):
            count = 0
            if d <= 26:
              count += (θ_i)*((2*pre_pixel[c][d + 1]) - 1) 
            if c >= 1:
              count += (θ_i)*((2*pre_pixel[c - 1][d]) - 1) 
            if c <= 26:
              count += (θ_i)*((2*pre_pixel[c + 1][d]) - 1) 
            if d >= 1:
              count += (θ_i)*((2*pre_pixel[c][d - 1]) - 1) 
            count += dataset[a][c][d]*(θ_j)
            exp_count = np.exp(count)
            compo = np.exp(-1*count) + np.exp(count) 
            pixel[c][d] = exp_count/compo
          pre_pixel = np.copy(pixel)
        denoise_img[a] = pixel
    denoise_img = np.asarray(denoise_img)
    return denoise_img

def second_binarize(dataset, dimone = 500, dimtwo = 28, dimthree = 28):
    pixel = np.zeros((dimone,dimtwo,dimthree))
    for i in range(dimone):
      for j in range(dimtwo):
        for k in range(dimthree):
          pixel[i][j][k] = -1 if dataset[i][j][k] <= 0.5 else 1
    return pixel

def calculate_fractions(datasetl,dataset2): 
    num = np.zeros(datasetl.shape[0]) 
    accuracy = np.copy(num)
    for i in range(datasetl.shape[0]):
      for j in range(datasetl.shape[1]):
        for k in range(datasetl.shape[2]):
          if datasetl[i][j][k] == dataset2[i][j][k]:
            num[i] += 1
      accuracy[i] = num[i]/(datasetl.shape[1]**2) 
    most_accuracy = num.argsort()[499]
    least_accuracy = num.argsort()[0]
    return accuracy, least_accuracy, most_accuracy

def display(img):
    plt.imshow(img, cmap = 'gray') 
    plt.show()

original = retrieve_images(x_train)  # obtain the original 500 MNIST images
original = binarize(original)         # binarize the original images

noise_version_image = copy.deepcopy(original)
noise_version_image = noise(noise_version_image, original) # creat a noisy version for each image

denoise_version_image = second_binarize(denoise(noise_version_image, 0.2, 0.2)) # create a denoise version for each image 

accuracies,least,most = calculate_fractions(denoise_version_image,original) # calculate the accuracies for the 500 images
print(sum(accuracies/len(accuracies))) # calculate the average accuracy over the 500 images

display(original[most])
display(noise_version_image[most])
display(denoise_version_image[most]) # display the original, noisy, and 
display(original[least])             # reconstruction for the most and 
display(noise_version_image[least]) # least accurate reconstruction 
display(denoise_version_image[least])
