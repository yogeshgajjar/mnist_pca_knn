########################################################################################################################
#                                   IMAGE CLASSIFICATION USING PCA & K-NN 
########################################################################################################################


import struct as st 
import numpy as np 
import sys
from sklearn.decomposition import PCA


def knn(train_images, test_images, train_labels, test_labels, K):
    for i in range(len(test_images)):  # loop for calculating distance of each query point with the training datapoint
        distances = np.zeros(len(train_images))
        knn_dist = np.zeros(K)
        result = np.zeros(K)
        total = 0.0
        weight_votes= np.zeros(10)
        for j in range(len(train_images)): # loop for iterating all the training datapoint
            distances[j] = np.sqrt(np.sum(np.square(test_images[i] - train_images[j])))  #calculting eucleadian distance 

        sort = distances.argsort()  # finding the index of the point having least distance of query point from all the nearby points
        for k in range(K):
            index = sort[k]
            knn_dist[k] = distances[index]  # K nearest neighbours 
            result[k] = 1.0 / knn_dist[k] #calculating weight which is inverse of distance
            total += result[k]  # adding the total weights 
            result /= total # and dividing it by the total 
            pre_label = train_labels[index]  # predicting the label with the datapoint having the maximum weight. 
            weight_votes[pre_label] += result[k] # Adding the weights to the predicted label to find the maximum weight
     

        f = open("finaloutput.txt", 'a+')
        f.write("%s %s\n" % (int(pre_label[0]), int(test_labels[i][0])))
        f.close()

def extract(N, path):  # code reference https://medium.com/@mannasiladittya/converting-mnist-data-in-idx-format-to-python-numpy-array-5cb9126f99f1 

    train_imagesfile = open(path + 'train-images.idx3-ubyte', 'rb')
    train_labelfile = open(path + 'train-labels.idx1-ubyte', 'rb')

    train_imagesfile.seek(0)
    magic_number_images = st.unpack('>4B',train_imagesfile.read(4))
    img = st.unpack('>I',train_imagesfile.read(4))[0] #num of images
    row = st.unpack('>I',train_imagesfile.read(4))[0] #num of rows
    col = st.unpack('>I',train_imagesfile.read(4))[0] #no of columns

    total_dim = img*row*col
    images_array = np.asarray(st.unpack('>'+'B'*total_dim,train_imagesfile.read(total_dim))).reshape((img,row,col))  # creating arrray from the mnist dataset by extracting image
     
    train_labelfile.seek(0) 
    magic_number_labels = st.unpack('>4B', train_labelfile.read(4))
    label = st.unpack('>I', train_labelfile.read(4))[0]
    labels_array = np.asarray(st.unpack('>'+'B'*label,train_labelfile.read(label))).reshape((label,1))  # creating arrray from the mnist dataset by extracting labels. 

    images = images_array[0:1000,:,:].reshape(1000,-1)  # columnizing it into 784x1 vector
    labels = labels_array[0:1000]

    test_images = images[0:N,:]  # Test & Training
    train_images = images[N:1000, :]  
    test_labels = labels[0:N]
    train_labels = labels[N:1000]

    return train_images, test_images, train_labels, test_labels

def pca(train_images, test_images, D):
    pca = PCA(n_components = D, svd_solver= 'full')
    pca.fit(train_images)

    train_images = pca.transform(train_images)
    test_images = pca.transform(test_images)

    return train_images, test_images



if len(sys.argv) == 5:
    K = int(sys.argv[1])
    D = int(sys.argv[2])
    N = int(sys.argv[3])
    path = str(sys.argv[4])


train_img, test_img, train_lab, test_lab = extract(N, path)
train_images_pca, test_images_pca = pca(train_img, test_img, D)
knn(train_images_pca,test_images_pca,train_lab, test_lab, K)