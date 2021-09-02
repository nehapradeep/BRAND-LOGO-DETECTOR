# importing necessary modules and packages
import numpy as np
import os
import cv2
import random
import pickle
from tqdm import tqdm
import random
from keras.utils import to_categorical


# directory of the data to be trained
DATADIR = "C:/Users/NANDITHA/Desktop/B.Tech/S8/AI/Project/data/training"

# four categories of data 
CATEGORIES = ["adidas", "nike" , "puma" , "reebok"]

# size of image to which it has to be resized
IMG_SIZE = 100


training_data = []

# function to create training data
def create_training_data():
    for category in CATEGORIES:  
        path = os.path.join(DATADIR,category)  # creating path to each category
        class_num = CATEGORIES.index(category)  # getting the label of each category
        encoded = to_categorical(class_num, num_classes=4)   # converting the labels to one hot encoding
        
        for img in tqdm(os.listdir(path)):  # iterating through all images of dataset
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  # reading the image in grayscale and converting to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize the image to normalize data size
                training_data.append([new_array, class_num])  # add this to the training_data
            except Exception as e:  # to avoid any errors
                pass
            
# calling the create training data function
create_training_data()

# printing the total no. of training data
print("No. of training data=",len(training_data))

# shuffling the data
random.shuffle(training_data) 


X = []
y = []

# storing the features in X and labels in y
for features,label in training_data:
    X.append(features)
    y.append(label)

# reshaping to store the data in 1D
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) 

# saving X and y datas on to disk to use in future
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

